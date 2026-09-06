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
        use_cache: bool = True,
    ):
        self.sequence_dir = Path(sequence_dir)
        self.dataset = SemanticKITTIDataset(
            self.sequence_dir
        )

        self.projector = (
            projector
            if projector is not None
            else BEVProjector()
        )
        
        # Setup Disk Cache to save hours of redundant compute
        self.use_cache = use_cache
        self.cache_dir = self.sequence_dir / ".bev_cache"
        if self.use_cache:
            self.cache_dir.mkdir(parents=True, exist_ok=True)

    def __len__(self) -> int:
        return len(self.dataset)

    def __getitem__(self, index: int):
        # 1. Generate a unique cache signature based on projector settings
        # This guarantees that if we change resolution/ranges, it ignores the old cache
        if self.use_cache:
            proj = self.projector
            config_str = f"res_{proj.resolution}_x_{proj.x_min}_{proj.x_max}_y_{proj.y_min}_{proj.y_max}_z_{proj.z_min}_{proj.z_max}"
            
            # Create a specific sub-folder for this exact configuration
            config_cache_dir = self.cache_dir / config_str
            config_cache_dir.mkdir(parents=True, exist_ok=True)
            
            cache_file = config_cache_dir / f"frame_{index:06d}.pt"
            
            if cache_file.exists():
                return torch.load(cache_file, weights_only=False)

        # 2. If not cached, do the heavy computation
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

        data = {
            "features": features,
            "target": target,
            "mask": mask,
        }
        
        # 3. Save to cache for the next epoch!
        if self.use_cache:
            torch.save(data, cache_file)

        return data
