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
        
        self.use_cache = use_cache

        # VERY IMPORTANT: Default to checking inside the sequence directory itself for .bev_cache
        # This allows pre-generated caches on Google Drive to be found instantly!
        import os
        import tempfile
        import hashlib
        
        # If the user explicitly provided a cache dir, use it. Otherwise, default to sequence_dir.
        env_cache = os.environ.get("BEV_CACHE_DIR")
        if env_cache:
            abs_path = str(self.sequence_dir.absolute()).encode('utf-8')
            path_hash = hashlib.md5(abs_path).hexdigest()[:8]
            self.cache_dir = Path(env_cache) / "bev_cache" / f"{self.sequence_dir.name}_{path_hash}"
        else:
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
            config_str = f"config_{proj.config_hash}"
            
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
