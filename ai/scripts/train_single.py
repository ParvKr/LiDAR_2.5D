import argparse
import logging
from pathlib import Path
import sys

# Add the 'ai' directory to Python path so internal imports work
sys.path.append(str(Path(__file__).resolve().parents[1]))

import torch
from torch.utils.data import DataLoader

from data.datasets.bev_dataset import BEVDataset
from losses.segmentation import masked_focal_loss
from models.unet import UNet


logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train a UNet model on BEV dataset.")
    parser.add_argument(
        "--sequence-dir",
        type=Path,
        default=Path(__file__).resolve().parents[2] / "SemanticKITTI/sequences/00",
        help="Path to SemanticKITTI sequence directory.",
    )
    parser.add_argument("--epochs", type=int, default=50, help="Number of training epochs.")
    parser.add_argument("--patience", type=int, default=10, help="Early stopping patience (epochs).")
    parser.add_argument("--learning-rate", type=float, default=1e-3, help="Learning rate.")
    return parser.parse_args()


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

    dataset = BEVDataset(
        args.sequence_dir
    )

    # 2. Optimize DataLoader: 
    # - Increased batch size from 1 to 4 (RTX 4050 can handle this easily)
    # - Set num_workers=4 for parallel disk IO
    # - Set pin_memory=True for faster CPU-to-GPU transfers
    loader = DataLoader(
        dataset,
        batch_size=4,
        shuffle=True,
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
        total_loss = 0.0

        for batch in loader:
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

            total_loss += loss.item()

        if epoch == 1 or epoch % 10 == 0:
            logger.info(
                f"Epoch {epoch:3d} | "
                f"Loss: {total_loss:.6f}"
            )

        if total_loss < best_loss:
            best_loss = total_loss
            epochs_without_improvement = 0
            # Save the best model!
            torch.save(model.state_dict(), "best_unet.pth")
            logger.info("New best model saved to best_unet.pth")
        else:
            epochs_without_improvement += 1
            if epochs_without_improvement >= args.patience:
                logger.info(f"Early stopping triggered after {epoch} epochs.")
                break


if __name__ == "__main__":
    main()
