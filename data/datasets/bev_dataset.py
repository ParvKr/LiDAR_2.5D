from pathlib import Path

import torch
from torch.utils.data import Dataset

from data.datasets.semantic_kitti import SemanticKITTIDataset
from geometry.bev import BEVProjector


class BEVDataset(Dataset):
    def __init__(
        self,
        sequence_dir: str | Path,
        projector: BEVProjector | None = None,
    ):
        self.dataset = SemanticKITTIDataset(
            sequence_dir
        )

        self.projector = (
            projector
            if projector is not None
            else BEVProjector()
        )

    def __len__(self) -> int:
        return len(self.dataset)

    def __getitem__(self, index: int):
        points, labels = self.dataset[index]

        result = self.projector.project(
            points,
            labels,
        )

        features = torch.from_numpy(
            result.features
        ).float()

        target = torch.from_numpy(
            result.labels
        ).long()

        mask = torch.from_numpy(
            result.label_mask
        ).bool()

        return {
            "features": features,
            "target": target,
            "mask": mask,
        }