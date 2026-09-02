from pathlib import Path

import torch

from data.datasets.bev_dataset import BEVDataset
from models.unet import UNet


SEQUENCE_PATH = Path(
    "../SemanticKITTI/sequences/00"
)


def main():
    dataset = BEVDataset(
        SEQUENCE_PATH
    )

    sample = dataset[0]

    x = sample["features"].unsqueeze(0)

    print("Input:")
    print(x.shape)

    model = UNet(
        in_channels=5,
        num_classes=20,
    )

    with torch.no_grad():
        output = model(x)

    print("\nOutput:")
    print(output.shape)

    print("\nExpected:")
    print("[1, 20, 500, 500]")


if __name__ == "__main__":
    main()