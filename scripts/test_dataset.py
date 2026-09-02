from pathlib import Path

import torch

from data.datasets.bev_dataset import BEVDataset


SEQUENCE_PATH = Path(
    "../SemanticKITTI/sequences/00"
)


def main():
    dataset = BEVDataset(
        SEQUENCE_PATH
    )

    print(f"Dataset size: {len(dataset)}")

    sample = dataset[0]

    features = sample["features"]
    target = sample["target"]
    mask = sample["mask"]

    print("\nSample:")
    print(
        f"Features: {features.shape} "
        f"{features.dtype}"
    )
    print(
        f"Target:   {target.shape} "
        f"{target.dtype}"
    )
    print(
        f"Mask:     {mask.shape} "
        f"{mask.dtype}"
    )

    print("\nValidation:")

    print(
        f"Features finite: "
        f"{torch.isfinite(features).all().item()}"
    )

    print(
        f"Target min: {target.min().item()}"
    )

    print(
        f"Target max: {target.max().item()}"
    )

    print(
        f"Trainable cells: "
        f"{mask.sum().item()}"
    )

    print(
        f"Masked target values: "
        f"{target[mask].unique().tolist()}"
    )


if __name__ == "__main__":
    main()