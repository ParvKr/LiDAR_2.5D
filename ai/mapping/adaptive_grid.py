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
        """Insert an Nx3 array of LiDAR points."""

        points = np.asarray(points, dtype=np.float32)

        if points.ndim != 2 or points.shape[1] != 3:
            raise ValueError(
                "points must have shape (N, 3)."
            )

        if semantic_classes is not None:
            semantic_classes = np.asarray(semantic_classes)
            if semantic_classes.shape != (len(points),):
                raise ValueError("semantic_classes must have shape (N,).")

        for index, (x, y, z) in enumerate(points):
            self.insert_point(
                x=x,
                y=y,
                z=z,
                semantic_class=(None if semantic_classes is None else semantic_classes[index]),
            )

    @property
    def num_cells(self) -> int:
        return len(self.cells)
