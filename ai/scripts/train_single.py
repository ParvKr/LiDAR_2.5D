import argparse
import logging
from pathlib import Path
import sys

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
        in_channels=5,
        num_classes=20,
    ).to(device)

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=args.learning_rate,
    )

    # 3. Initialize AMP Scaler for Mixed Precision Training
    scaler = torch.amp.GradScaler(device.type) if device.type == 'cuda' else None

    model.train()

    best_loss = float("inf")
    epochs_without_improvement = 0

    for epoch in range(1, args.epochs + 1):
        # --- TRAINING ---
        model.train()
        train_loss = 0.0

        for batch in train_loader:
            features = batch["features"].to(device)
            target = batch["target"].to(device)
            mask = batch["mask"].to(device)

            optimizer.zero_grad()

            # 4. Mixed Precision Training (FP16)
            if device.type == "cuda":
                with torch.amp.autocast('cuda'):
                    logits = model(features)
                    loss = masked_focal_loss(logits, target, mask)
                
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
            else:
                # Standard FP32 for CPU/MPS
                logits = model(features)
                loss = masked_focal_loss(logits, target, mask)
                loss.backward()
                optimizer.step()

            train_loss += loss.item()

        # --- VALIDATION ---
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for batch in val_loader:
                features = batch["features"].to(device)
                target = batch["target"].to(device)
                mask = batch["mask"].to(device)

                if device.type == "cuda":
                    with torch.amp.autocast('cuda'):
                        logits = model(features)
                        loss = masked_focal_loss(logits, target, mask)
                else:
                    logits = model(features)
                    loss = masked_focal_loss(logits, target, mask)
                
                val_loss += loss.item()

        # Average the losses over the number of batches
        train_loss /= max(1, len(train_loader))
        val_loss /= max(1, len(val_loader))

        logger.info(
            f"Epoch {epoch:3d} | "
            f"Train Loss: {train_loss:.6f} | "
            f"Val Loss: {val_loss:.6f}"
        )

        # Early Stopping based on Validation Loss
        if val_loss < best_loss:
            best_loss = val_loss
            epochs_without_improvement = 0
            
            # Save the best model safely to the checkpoint directory!
            args.checkpoint_dir.mkdir(parents=True, exist_ok=True)
            best_model_path = args.checkpoint_dir / "best_unet.pth"
            torch.save(model.state_dict(), best_model_path)
            logger.info(f"New best model saved to {best_model_path}")
        else:
            epochs_without_improvement += 1
            if epochs_without_improvement >= args.patience:
                logger.info(f"Early stopping triggered after {epoch} epochs.")
                break


if __name__ == "__main__":
    main()
