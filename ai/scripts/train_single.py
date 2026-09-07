import argparse
import logging
from pathlib import Path
import sys
from tqdm import tqdm

# Add the 'ai' directory to Python path so internal imports work
sys.path.append(str(Path(__file__).resolve().parents[1]))

import torch
from torch.utils.data import DataLoader, ConcatDataset

from data.datasets.bev_dataset import BEVDataset
from losses.segmentation import masked_focal_loss
from models.unet import UNet


logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train a UNet model on BEV dataset.")
    parser.add_argument(
        "--dataset-root",
        type=Path,
        default=Path(__file__).resolve().parents[2] / "SemanticKITTI",
        help="Path to SemanticKITTI root directory.",
    )
    parser.add_argument("--train-seqs", nargs="+", default=["00","01","02","03","04","05","06","07","09"], help="Training sequences")
    parser.add_argument("--val-seqs", nargs="+", default=["08","10"], help="Validation sequences")
    parser.add_argument("--checkpoint-dir", type=Path, default=Path("checkpoints"), help="Directory to save best model.")
    parser.add_argument("--epochs", type=int, default=50, help="Number of training epochs.")
    parser.add_argument("--patience", type=int, default=10, help="Early stopping patience (epochs).")
    parser.add_argument("--learning-rate", type=float, default=1e-3, help="Learning rate.")
    return parser.parse_args()


def get_concat_dataset(root: Path, seqs: list[str]) -> ConcatDataset:
    datasets = []
    for seq in seqs:
        seq_dir = root / "sequences" / seq
        if seq_dir.exists():
            datasets.append(BEVDataset(seq_dir))
        else:
            logger.warning(f"Sequence {seq} not found at {seq_dir}, skipping.")
    return ConcatDataset(datasets)

def compute_class_weights(dataset_root: Path, seqs: list[str], num_classes: int = 20) -> torch.Tensor:
    from perception.taxonomy import LEARNING_MAP
    import numpy as np

    logger.info("Computing inverse-frequency class weights from raw labels...")
    
    # Fast vectorized mapping using a lookup array for 16-bit semantic IDs
    mapping_array = np.zeros(65536, dtype=np.int32)
    for k, v in LEARNING_MAP.items():
        mapping_array[k] = v

    class_counts = np.zeros(num_classes, dtype=np.int64)
    
    for seq in seqs:
        labels_dir = dataset_root / "sequences" / seq / "labels"
        if not labels_dir.exists():
            continue
            
        for label_file in labels_dir.glob("*.label"):
            raw_labels = np.fromfile(label_file, dtype=np.uint32)
            semantic_ids = raw_labels & 0xFFFF
            mapped = mapping_array[semantic_ids]
            counts = np.bincount(mapped, minlength=num_classes)
            class_counts += counts

    # Calculate inverse frequencies
    class_counts[0] = 0  # Ignore class 0 (unlabeled)
    total_valid = max(class_counts.sum(), 1)
    freq = class_counts / total_valid
    
    weights = np.ones(num_classes, dtype=np.float32)
    valid_mask = freq > 0
    weights[valid_mask] = 1.0 / (freq[valid_mask] + 1e-6)
    
    # Clamp weights between ~0.2x and ~2.0x to avoid extreme loss oscillation
    if valid_mask.any():
        q10 = np.percentile(weights[valid_mask], 10)
        q90 = np.percentile(weights[valid_mask], 90)
        weights = np.clip(weights, q10, q90)
        # Normalize so the mean weight of valid classes is exactly 1.0
        weights[valid_mask] = weights[valid_mask] / weights[valid_mask].mean()
        
    weights[0] = 0.0 # Force unlabeled class to 0 completely
    
    logger.info(f"Computed Class Weights: \n{weights}")
    return torch.from_numpy(weights)


def main():
    args = parse_args()

    device = torch.device(
        "cuda"
        if torch.cuda.is_available()
        else "cpu"
    )
    # 1. Enable cuDNN benchmark for faster convolutions on fixed-size inputs
    if device.type == "cuda":
        torch.backends.cudnn.benchmark = True

    logger.info(f"Using device: {device}")

    train_dataset = get_concat_dataset(args.dataset_root, args.train_seqs)
    val_dataset = get_concat_dataset(args.dataset_root, args.val_seqs)

    # 2. Optimize DataLoader: 
    train_loader = DataLoader(
        train_dataset,
        batch_size=4,
        shuffle=True,
        num_workers=4,
        pin_memory=(device.type == "cuda"),
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=4,
        shuffle=False,
        num_workers=4,
        pin_memory=(device.type == "cuda"),
    )

    model = UNet(
        in_channels=6,
        num_classes=20,
    ).to(device)

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=args.learning_rate,
    )

    # 3. Learning Rate Scheduler (Reduces LR when validation loss plateaus)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=0.5, patience=3
    )

    # 4. Initialize AMP Scaler for Mixed Precision Training
    scaler = torch.amp.GradScaler(device.type) if device.type == 'cuda' else None

    # 5. Per-Class Focal Loss Weights (Alpha)
    # Computed dynamically from raw labels to perfectly combat class imbalance
    class_weights = compute_class_weights(args.dataset_root, args.train_seqs).to(device)

    model.train()

    best_miou = -1.0
    epochs_without_improvement = 0

    for epoch in range(1, args.epochs + 1):
        # --- TRAINING ---
        model.train()
        train_loss = 0.0

        # Wrap the dataloader in tqdm for a progress bar
        train_pbar = tqdm(train_loader, desc=f"Epoch {epoch}/{args.epochs} [Train]")
        
        for batch in train_pbar:
            # We cast features up to float32 (AMP will handle casting down if safe)
            # and target to long (int64) because F.cross_entropy requires it.
            features = batch["features"].to(device, dtype=torch.float32)
            target = batch["target"].to(device, dtype=torch.long)
            mask = batch["mask"].to(device)

            optimizer.zero_grad()

            # Mixed Precision Training (FP16)
            if device.type == "cuda":
                with torch.amp.autocast('cuda'):
                    logits = model(features)
                    loss = masked_focal_loss(logits, target, mask, alpha=class_weights)
                
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
            else:
                # Standard FP32 for CPU/MPS
                logits = model(features)
                loss = masked_focal_loss(logits, target, mask, alpha=class_weights)
                loss.backward()
                optimizer.step()

            train_loss += loss.item()
            train_pbar.set_postfix({"loss": f"{loss.item():.4f}"})

        # --- VALIDATION ---
        model.eval()
        val_loss = 0.0
        
        intersection = torch.zeros(20, device=device)
        union = torch.zeros(20, device=device)
        
        val_pbar = tqdm(val_loader, desc=f"Epoch {epoch}/{args.epochs} [Val]")
        
        with torch.no_grad():
            for batch in val_pbar:
                features = batch["features"].to(device, dtype=torch.float32)
                target = batch["target"].to(device, dtype=torch.long)
                mask = batch["mask"].to(device)

                if device.type == "cuda":
                    with torch.amp.autocast('cuda'):
                        logits = model(features)
                        loss = masked_focal_loss(logits, target, mask, alpha=class_weights)
                else:
                    logits = model(features)
                    loss = masked_focal_loss(logits, target, mask, alpha=class_weights)
                
                val_loss += loss.item()
                
                # Compute IoU components (ignore class 0)
                preds = logits.argmax(dim=1)
                valid = mask & (target != 0)
                
                for cls_id in range(1, 20):
                    cls_preds = (preds == cls_id) & valid
                    cls_target = (target == cls_id) & valid
                    
                    intersection[cls_id] += (cls_preds & cls_target).sum()
                    union[cls_id] += (cls_preds | cls_target).sum()
                    
                val_pbar.set_postfix({"loss": f"{loss.item():.4f}"})

        # Average the losses over the number of batches
        train_loss /= max(1, len(train_loader))
        val_loss /= max(1, len(val_loader))
        
        # Calculate Validation mIoU
        ious = intersection[1:] / (union[1:] + 1e-6)
        valid_classes = union[1:] > 0
        val_miou = ious[valid_classes].mean().item() if valid_classes.any() else float('-inf')

        # Update the Learning Rate Scheduler
        scheduler.step(val_loss)
        current_lr = optimizer.param_groups[0]['lr']

        logger.info(
            f"Epoch {epoch:3d} | "
            f"LR: {current_lr:.2e} | "
            f"Train Loss: {train_loss:.6f} | "
            f"Val Loss: {val_loss:.6f} | "
            f"Val mIoU: {val_miou:.4f}"
        )

        # Early Stopping based on Validation mIoU
        if val_miou > best_miou:
            best_miou = val_miou
            epochs_without_improvement = 0
            
            # Save the best model safely to the checkpoint directory!
            args.checkpoint_dir.mkdir(parents=True, exist_ok=True)
            best_model_path = args.checkpoint_dir / "best_unet.pth"
            torch.save(model.state_dict(), best_model_path)
            logger.info(f"New best model saved with mIoU {best_miou:.4f} to {best_model_path}")
        else:
            epochs_without_improvement += 1
            if epochs_without_improvement >= args.patience:
                logger.info(f"Early stopping triggered after {epoch} epochs.")
                break


if __name__ == "__main__":
    main()
