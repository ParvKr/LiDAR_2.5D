from dataclasses import dataclass

import numpy as np

@dataclass
class BEVResult:
    features: np.ndarray
    labels: np.ndarray
    point_to_cell: np.ndarray
    valid_mask: np.ndarray
    label_mask: np.ndarray
    point_count: np.ndarray

class BEVProjector:
    def __init__(
        self,
        x_range: tuple[float, float] = (-50.0, 50.0),
        y_range: tuple[float, float] = (-50.0, 50.0),
        z_range: tuple[float, float] = (-10.0, 35.0),
        resolution: float = 0.20,
        min_distance: float = 1.0,
    ):
        if x_range[0] >= x_range[1]:
            raise ValueError("x_range must be increasing")

        if y_range[0] >= y_range[1]:
            raise ValueError("y_range must be increasing")

        if z_range[0] >= z_range[1]:
            raise ValueError("z_range must be increasing")

        if resolution <= 0:
            raise ValueError(
                "resolution must be greater than zero"
            )

        if min_distance < 0:
            raise ValueError(
                "min_distance cannot be negative"
            )

        self.x_min, self.x_max = x_range
        self.y_min, self.y_max = y_range

        self.z_min, self.z_max = z_range

        self.resolution = resolution
        self.min_distance = min_distance

        self.width = int(
            np.ceil(
                (self.x_max - self.x_min)
                / self.resolution
            )
        )

        self.height = int(
            np.ceil(
                (self.y_max - self.y_min)
                / self.resolution
            )
        )

    @property
    def config_hash(self) -> str:
        import hashlib
        import json
        config = {
            "resolution": self.resolution,
            "x_min": self.x_min,
            "x_max": self.x_max,
            "y_min": self.y_min,
            "y_max": self.y_max,
            "z_min": self.z_min,
            "z_max": self.z_max,
            "min_distance": self.min_distance,
            "feature_version": 2, # Bumped because z_range filtering was fixed!
            "taxonomy_version": 1,
        }
        return hashlib.sha256(json.dumps(config, sort_keys=True).encode()).hexdigest()[:16]

    def project(
        self,
        points: np.ndarray,
        labels: np.ndarray,
    ) -> BEVResult:

        if points.ndim != 2 or points.shape[1] != 4:
            raise ValueError(
                "points must have shape (N, 4)"
            )

        if labels.ndim != 1:
            raise ValueError(
                "labels must have shape (N,)"
            )

        if len(points) != len(labels):
            raise ValueError(
                "points and labels must have the same length"
            )

        xyz = points[:, :3]
        intensity = points[:, 3]

        x = xyz[:, 0]
        y = xyz[:, 1]
        z = xyz[:, 2]

        distance = np.hypot(x, y)

        inside = (
            (x >= self.x_min)
            & (x < self.x_max)
            & (y >= self.y_min)
            & (y < self.y_max)
            & (z >= self.z_min)
            & (z < self.z_max)
            & (distance >= self.min_distance)
        )

        point_indices = np.flatnonzero(inside)

        x = x[inside]
        y = y[inside]
        z = z[inside]
        intensity = intensity[inside]
        labels = labels[inside]

        cols = (
            (x - self.x_min)
            / self.resolution
        ).astype(np.int32)

        rows = (
            (y - self.y_min)
            / self.resolution
        ).astype(np.int32)

        cell_ids = (
            rows * self.width
            + cols
        )

        num_cells = (
            self.height
            * self.width
        )

        max_height = np.full(
            num_cells,
            self.z_min,
            dtype=np.float32,
        )

        min_height = np.full(
            num_cells,
            self.z_max,
            dtype=np.float32,
        )

        sum_height = np.zeros(
            num_cells,
            dtype=np.float32,
        )

        sum_height_squared = np.zeros(
            num_cells,
            dtype=np.float32,
        )

        sum_intensity = np.zeros(
            num_cells,
            dtype=np.float32,
        )

        count = np.zeros(
            num_cells,
            dtype=np.int32,
        )

        np.maximum.at(
            max_height,
            cell_ids,
            z,
        )

        np.minimum.at(
            min_height,
            cell_ids,
            z,
        )

        np.add.at(
            sum_height,
            cell_ids,
            z,
        )

        np.add.at(
            sum_height_squared,
            cell_ids,
            z**2,
        )

        np.add.at(
            sum_intensity,
            cell_ids,
            intensity,
        )

        np.add.at(
            count,
            cell_ids,
            1,
        )

        occupied = count > 0

        mean_height = np.zeros(
            num_cells,
            dtype=np.float32,
        )
        
        height_variance = np.zeros(
            num_cells,
            dtype=np.float32,
        )

        mean_intensity = np.zeros(
            num_cells,
            dtype=np.float32,
        )

        mean_height[occupied] = (
            sum_height[occupied]
            / count[occupied]
        )
        
        height_variance[occupied] = np.maximum(
            0.0,
            (sum_height_squared[occupied] / count[occupied]) - mean_height[occupied]**2
        )

        mean_intensity[occupied] = (
            sum_intensity[occupied]
            / count[occupied]
        )

        density = np.log1p(
            count
        ).astype(np.float32)

        height_range = (
                self.z_max
                - self.z_min
        )

        max_height = (
                (max_height - self.z_min)
                / height_range
        )

        min_height = (
                (min_height - self.z_min)
                / height_range
        )

        mean_height = (
                (mean_height - self.z_min)
                / height_range
        )
        
        # Normalize variance (variance has units of m^2)
        height_variance = height_variance / (height_range ** 2)

        max_height[~occupied] = 0.0
        min_height[~occupied] = 0.0
        mean_height[~occupied] = 0.0
        height_variance[~occupied] = 0.0

        features = np.stack(
            [
                max_height,
                min_height,
                mean_height,
                height_variance,
                mean_intensity,
                density,
            ],
            axis=0,
        )

        features = features.reshape(
            6,
            self.height,
            self.width,
        )

        cell_labels = np.zeros(
            num_cells,
            dtype=np.int32,
        )

        # --- FAST MAJORITY VOTING (Fully Vectorized) ---
        if len(cell_ids) > 0:
            max_label = labels.max()
            
            # Create a unique 1D index for every (cell, label) pair
            flat_indices = cell_ids * (max_label + 1) + labels
            
            # Count occurrences purely in C using bincount
            counts = np.bincount(flat_indices, minlength=num_cells * (max_label + 1))
            
            # Reshape back to [num_cells, max_label + 1] and find the max
            counts_2d = counts.reshape(num_cells, max_label + 1)
            cell_labels = np.argmax(counts_2d, axis=1).astype(np.int32)

        cell_labels = cell_labels.reshape(
            self.height,
            self.width,
        )

        valid_mask = occupied.reshape(
            self.height,
            self.width,
        )

        label_mask = (
                valid_mask
                & (cell_labels != 0)
        )

        point_to_cell = np.full(
            len(points),
            -1,
            dtype=np.int32,
        )

        point_to_cell[
            point_indices
        ] = cell_ids

        point_count = count.reshape(
            self.height,
            self.width,
        )

        return BEVResult(
            features=features,
            labels=cell_labels,
            point_to_cell=point_to_cell,
            valid_mask=valid_mask,
            label_mask=label_mask,
            point_count=point_count,
        )