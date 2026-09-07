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
        
        # Setup Disk Cache
        self.use_cache = use_cache
        
        # VERY IMPORTANT: We save the cache to a fast local temporary directory 
        # instead of inside the sequence_dir. If sequence_dir is on a network drive 
        # (like Google Drive in Colab), writing 4,000 files will take 6+ hours due to sync overhead!
        import os
        import tempfile
        base_cache_dir = Path(os.environ.get("BEV_CACHE_DIR", tempfile.gettempdir())) / "bev_cache"
        self.cache_dir = base_cache_dir / self.sequence_dir.name
        
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
                data = torch.load(cache_file, weights_only=False)
                # CRITICAL: We must cast these back up to float32 and int64 IMMEDIATELY upon loading.
                # If we have a mix of old files (float32/int64) and new files (float16/int8) on disk,
                # the PyTorch DataLoader will crash when trying to batch them together unless they
                # are all standardized here first!
                data["features"] = data["features"].to(torch.float32)
                data["target"] = data["target"].to(torch.long)
                return data

        # 2. If not cached, do the heavy computation
        points, labels = self.dataset[index]

        result = self.projector.project(
            points,
            labels,
        )

        features = torch.from_numpy(
            result.features
        ).to(torch.float32)

        target = torch.from_numpy(
            result.labels
        ).to(torch.long)

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
            # Downcast to save 60% disk space ONLY for the saved file
            cached_data = {
                "features": features.to(torch.float16),
                "target": target.to(torch.int8),
                "mask": mask
            }
            torch.save(cached_data, cache_file)

        return data
