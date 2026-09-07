from pathlib import Path

import torch

from data.datasets.bev_dataset import BEVDataset
from losses.segmentation import masked_focal_loss
from models.unet import UNet


SEQUENCE_PATH = Path(__file__).resolve().parents[2] / "SemanticKITTI/sequences/00"


def main():
    dataset = BEVDataset(
        SEQUENCE_PATH
    )

    sample = dataset[0]

    features = sample["features"].unsqueeze(0)
    target = sample["target"].unsqueeze(0)
    mask = sample["mask"].unsqueeze(0)

    model = UNet(
        in_channels=6,
        num_classes=20,
    )

    logits = model(features)

    loss = masked_focal_loss(
        logits,
        target,
        mask,
    )

    print(f"Logits: {logits.shape}")
    print(f"Target: {target.shape}")
    print(f"Mask:   {mask.shape}")

    print(
        f"\nTrainable cells: "
        f"{mask.sum().item()}"
    )

    print(
        f"Loss: {loss.item():.6f}"
    )

    print(
        f"Loss finite: "
        f"{torch.isfinite(loss).item()}"
    )


if __name__ == "__main__":
    main()
