from typing import Dict, Optional, Tuple

import numpy as np

from mapping.cell import MapCell


class AdaptiveGrid:
    """
    Distance-adaptive 2.5D grid.

    Cell resolution increases with horizontal distance
    from the LiDAR sensor.
    """

    def __init__(
        self,
        resolution_bands: list[tuple[float, float]],
    ):
        if not resolution_bands:
            raise ValueError(
                "resolution_bands cannot be empty."
            )

        previous_max_distance = 0.0
        for max_distance, resolution in resolution_bands:
            if max_distance <= 0:
                raise ValueError(
                    "Maximum distance must be positive."
                )

            if resolution <= 0:
                raise ValueError(
                    "Resolution must be positive."
                )
            if max_distance <= previous_max_distance:
                raise ValueError(
                    "resolution_bands must have strictly increasing maximum distances."
                )
            previous_max_distance = max_distance

        self.resolution_bands = tuple(
            (float(max_distance), float(resolution))
            for max_distance, resolution in resolution_bands
        )

        self.cells: Dict[
            Tuple[int, int, int],
            MapCell
        ] = {}

    def get_band_index(self, distance: float) -> Optional[int]:
        """
        Return the horizontal resolution for a distance.
        """

        if distance < 0:
            raise ValueError(
                "distance cannot be negative."
            )

        for index, (max_distance, _) in enumerate(self.resolution_bands):
            if distance <= max_distance:
                return index

        return None

    def get_resolution(self, distance: float) -> float:
        """Return the configured resolution, or reject out-of-range points."""

        band_index = self.get_band_index(distance)
        if band_index is None:
            raise ValueError(
                f"distance {distance} exceeds map range {self.max_distance}."
            )
        return self.resolution_bands[band_index][1]

    @property
    def max_distance(self) -> float:
        return self.resolution_bands[-1][0]

    def point_to_index(
        self,
        x: float,
        y: float,
    ) -> Tuple[int, int, int]:
        """Convert a point into its adaptive grid index."""

        distance = float(np.hypot(x, y))

        band_index = self.get_band_index(distance)
        if band_index is None:
            raise ValueError(
                f"Point ({x}, {y}) lies outside the {self.max_distance} m map range."
            )
        resolution = self.resolution_bands[band_index][1]

        row = int(np.floor(y / resolution))
        col = int(np.floor(x / resolution))

        return row, col, band_index

    def get_or_create_cell(
        self,
        row: int,
        col: int,
        band_index: int,
    ) -> MapCell:
        """Return an existing cell or create one."""

        key = (row, col, band_index)

        if key not in self.cells:
            self.cells[key] = MapCell()

        return self.cells[key]

    def insert_point(
        self,
        x: float,
        y: float,
        z: float,
        semantic_class: int | None = None,
    ) -> MapCell:
        """Insert one LiDAR point into the adaptive grid."""

        row, col, band_index = self.point_to_index(x, y)

        cell = self.get_or_create_cell(
            row=row,
            col=col,
            band_index=band_index,
        )

        cell.add_observation(
            height=z,
            semantic_class=semantic_class,
        )

        return cell

    def insert_points(
        self,
        points: np.ndarray,
        semantic_classes: np.ndarray | None = None,
    ) -> None:
        """Vectorized insertion of an Nx3 array of LiDAR points."""
        points = np.asarray(points, dtype=np.float32)

        if points.ndim != 2 or points.shape[1] != 3:
            raise ValueError("points must have shape (N, 3).")

        if len(points) == 0:
            return

        x = points[:, 0]
        y = points[:, 1]
        z = points[:, 2]
        
        distances = np.hypot(x, y)
        
        # 1. Determine bands
        band_indices = np.full(len(points), -1, dtype=np.int32)
        resolutions = np.zeros(len(points), dtype=np.float32)
        
        for idx, (max_dist, res) in enumerate(self.resolution_bands):
            prev = self.resolution_bands[idx-1][0] if idx > 0 else -1.0
            mask = (distances > prev) & (distances <= max_dist)
            band_indices[mask] = idx
            resolutions[mask] = res
            
        # Filter out points beyond max distance
        valid = band_indices >= 0
        x = x[valid]
        y = y[valid]
        z = z[valid]
        band_indices = band_indices[valid]
        resolutions = resolutions[valid]
        
        if semantic_classes is not None:
            semantic_classes = semantic_classes[valid]
            
        if len(x) == 0:
            return

        # 2. Compute rows and cols
        rows = np.floor(y / resolutions).astype(np.int64)
        cols = np.floor(x / resolutions).astype(np.int64)
        
        # 3. Pack keys for np.unique (Shift by 20000 to handle negative indices safely)
        r_shifted = rows + 20000
        c_shifted = cols + 20000
        b_shifted = band_indices.astype(np.int64)
        
        packed_keys = (r_shifted << 32) | (c_shifted << 16) | b_shifted
        
        unique_keys, inverse = np.unique(packed_keys, return_inverse=True)
        
        num_unique = len(unique_keys)
        
        sum_height = np.zeros(num_unique, dtype=np.float64)
        sum_height_sq = np.zeros(num_unique, dtype=np.float64)
        point_counts = np.zeros(num_unique, dtype=np.int32)
        
        np.add.at(sum_height, inverse, z)
        np.add.at(sum_height_sq, inverse, z**2)
        np.add.at(point_counts, inverse, 1)
        
        if semantic_classes is not None:
            max_class = 256 # Supports UNSEEN = 255
            flat_indices = inverse * max_class + semantic_classes
            class_counts = np.bincount(flat_indices, minlength=num_unique * max_class)
            class_counts = class_counts.reshape(num_unique, max_class)
        
        # 4. Populate the dictionary directly
        for i, u_key in enumerate(unique_keys):
            b = int(u_key & 0xFFFF)
            c = int((u_key >> 16) & 0xFFFF) - 20000
            r = int((u_key >> 32) & 0xFFFFFFFF) - 20000
            
            cell = self.get_or_create_cell(r, c, b)
            cell._height_sum += float(sum_height[i])
            cell._height_squared_sum += float(sum_height_sq[i])
            cell.point_count += int(point_counts[i])
            cell.occupied = True
            
            if semantic_classes is not None:
                # Merge the local counts into the cell's global _semantic_counts
                local_counts = class_counts[i]
                nonzero_classes = np.nonzero(local_counts)[0]
                
                for cls_idx in nonzero_classes:
                    count = int(local_counts[cls_idx])
                    cell._semantic_counts[int(cls_idx)] = cell._semantic_counts.get(int(cls_idx), 0) + count
                    
                cell._labeled_point_count += int(local_counts.sum())
                
                if cell._labeled_point_count > 0:
                    # Recompute global majority from the merged dictionary
                    cell.semantic_class = max(cell._semantic_counts, key=cell._semantic_counts.get)
                    cell.semantic_confidence = float(cell._semantic_counts[cell.semantic_class] / cell._labeled_point_count)

    @property
    def num_cells(self) -> int:
        return len(self.cells)
