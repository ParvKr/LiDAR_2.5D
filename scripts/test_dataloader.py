from pathlib import Path

import torch
from torch.utils.data import DataLoader

from data.datasets.bev_dataset import BEVDataset


SEQUENCE_PATH = Path(
    "../SemanticKITTI/sequences/00"
)


def main():
    dataset = BEVDataset(
        SEQUENCE_PATH
    )

    loader = DataLoader(
        dataset,
        batch_size=1,
        shuffle=True,
        num_workers=0,
    )

    batch = next(iter(loader))

    features = batch["features"]
    target = batch["target"]
    mask = batch["mask"]

    print("Batch:")

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

    print("\nExpected:")

    print(
        "Features: [batch, 5, 500, 500]"
    )

    print(
        "Target:   [batch, 500, 500]"
    )

    print(
        "Mask:     [batch, 500, 500]"
    )


if __name__ == "__main__":
    main()