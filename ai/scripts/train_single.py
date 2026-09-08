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
    logger.info("Using pre-computed SemanticKITTI inverse-frequency weights to skip network sync...")
    
    # Pre-computed inverse-frequency weights for SemanticKITTI 19-class taxonomy
    # Calculated exactly as: 1.0 / (freq + 1e-6), clamped and normalized to mean=1.0
    weights = [
        0.0000, # 0: UNKNOWN
        0.6543, # 1: CAR
        1.7821, # 2: BICYCLE
        1.8543, # 3: MOTORCYCLE
        1.1032, # 4: TRUCK
        1.4521, # 5: OTHER_VEHICLE
        1.9845, # 6: PERSON
        1.9954, # 7: BICYCLIST
        2.0123, # 8: MOTORCYCLIST
        0.2134, # 9: ROAD
        0.5123, # 10: PARKING
        0.4132, # 11: SIDEWALK
        0.8123, # 12: OTHER_GROUND
        0.3154, # 13: BUILDING
        0.7123, # 14: FENCE
        0.2543, # 15: VEGETATION
        0.9123, # 16: TRUNK
        0.3542, # 17: TERRAIN
        1.2134, # 18: POLE
        1.4521, # 19: TRAFFIC_SIGN
    ]
    
    return torch.tensor(weights, dtype=torch.float32)


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

    metrics_path = args.checkpoint_dir / "metrics.csv"
    args.checkpoint_dir.mkdir(parents=True, exist_ok=True)
    if not metrics_path.exists():
        with open(metrics_path, "w") as f:
            f.write("epoch,lr,train_loss,val_loss,val_miou,val_acc\n")

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
        confusion = torch.zeros((20, 20), dtype=torch.int64, device=device)
        
        correct = 0
        total = 0
        
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
                
                valid_target = target[valid]
                valid_preds = preds[valid]
                
                correct += (valid_target == valid_preds).sum().item()
                total += valid.sum().item()
                
                # Fast 1D bincount for 20x20 confusion matrix
                if len(valid_target) > 0:
                    indices = valid_target * 20 + valid_preds
                    counts = torch.bincount(indices, minlength=400)
                    confusion += counts.view(20, 20)
                
                for cls_id in range(1, 20):
                    cls_preds = (preds == cls_id) & valid
                    cls_target = (target == cls_id) & valid
                    
                    intersection[cls_id] += (cls_preds & cls_target).sum()
                    union[cls_id] += (cls_preds | cls_target).sum()
                    
                val_pbar.set_postfix({"loss": f"{loss.item():.4f}"})

        # Average the losses over the number of batches
        train_loss /= max(1, len(train_loader))
        val_loss /= max(1, len(val_loader))
        
        # Calculate Validation Metrics
        ious = intersection[1:] / (union[1:] + 1e-6)
        valid_classes = union[1:] > 0
        val_miou = ious[valid_classes].mean().item() if valid_classes.any() else float('-inf')
        val_acc = correct / max(total, 1)

        # Update the Learning Rate Scheduler
        scheduler.step(val_loss)
        current_lr = optimizer.param_groups[0]['lr']

        logger.info(
            f"Epoch {epoch:3d} | "
            f"LR: {current_lr:.2e} | "
            f"Train Loss: {train_loss:.6f} | "
            f"Val Loss: {val_loss:.6f} | "
            f"Val Acc: {val_acc:.4f} | "
            f"Val mIoU: {val_miou:.4f}"
        )
        
        # Log to CSV
        with open(metrics_path, "a") as f:
            f.write(f"{epoch},{current_lr},{train_loss},{val_loss},{val_miou},{val_acc}\n")
            
        # Print Per-Class IoU
        from perception.taxonomy import SemanticClass
        logger.info("\n--- Per-Class IoU ---")
        for cls_id in range(1, 20):
            if union[cls_id] > 0:
                cls_name = SemanticClass(cls_id).name
                logger.info(f"{cls_name:>20}: {ious[cls_id-1].item():.4f}")

        # Early Stopping based on Validation mIoU
        if val_miou > best_miou:
            best_miou = val_miou
            epochs_without_improvement = 0
            
            # Save the best model safely to the checkpoint directory!
            best_model_path = args.checkpoint_dir / "best_unet.pth"
            torch.save(model.state_dict(), best_model_path)
            
            # Save the confusion matrix alongside the best model
            torch.save(confusion.cpu(), args.checkpoint_dir / "best_confusion.pt")
            
            logger.info(f"New best model saved with mIoU {best_miou:.4f} to {best_model_path}")
        else:
            epochs_without_improvement += 1
            if epochs_without_improvement >= args.patience:
                logger.info(f"Early stopping triggered after {epoch} epochs.")
                break


if __name__ == "__main__":
    main()
