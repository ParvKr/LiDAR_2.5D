from pathlib import Path

import torch
from torch.utils.data import DataLoader

from data.datasets.bev_dataset import BEVDataset
from losses.segmentation import masked_cross_entropy
from models.unet import UNet


SEQUENCE_PATH = Path(
    "../SemanticKITTI/sequences/00"
)

EPOCHS = 100
LEARNING_RATE = 1e-3


def main():
    device = torch.device(
        "cuda"
        if torch.cuda.is_available()
        else "cpu"
    )

    print(f"Device: {device}")

    dataset = BEVDataset(
        SEQUENCE_PATH
    )

    loader = DataLoader(
        dataset,
        batch_size=1,
        shuffle=False,
        num_workers=0,
    )

    model = UNet(
        in_channels=5,
        num_classes=20,
    ).to(device)

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=LEARNING_RATE,
    )

    model.train()

    for epoch in range(1, EPOCHS + 1):
        total_loss = 0.0

        for batch in loader:
            features = batch["features"].to(device)
            target = batch["target"].to(device)
            mask = batch["mask"].to(device)

            optimizer.zero_grad()

            logits = model(features)

            loss = masked_cross_entropy(
                logits,
                target,
                mask,
            )

            loss.backward()

            optimizer.step()

            total_loss += loss.item()

        if epoch == 1 or epoch % 10 == 0:
            print(
                f"Epoch {epoch:3d} | "
                f"Loss: {total_loss:.6f}"
            )


if __name__ == "__main__":
    main()