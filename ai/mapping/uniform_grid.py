from typing import Dict, Tuple

import numpy as np

from mapping.cell import MapCell


class UniformGrid:
    """
    Uniform-resolution 2.5D grid.

    Every cell has the same horizontal resolution.
    """

    def __init__(
        self,
        resolution: float,
        x_min: float,
        y_min: float,
    ):
        if resolution <= 0:
            raise ValueError("resolution must be positive.")

        self.resolution = float(resolution)
        self.x_min = float(x_min)
        self.y_min = float(y_min)

        self.cells: Dict[Tuple[int, int], MapCell] = {}

    def point_to_index(
        self,
        x: float,
        y: float,
    ) -> Tuple[int, int]:
        """Convert metric x,y coordinates into a grid index."""

        col = int(
            np.floor((x - self.x_min) / self.resolution)
        )
        row = int(
            np.floor((y - self.y_min) / self.resolution)
        )

        return row, col

    def get_or_create_cell(
        self,
        row: int,
        col: int,
    ) -> MapCell:
        """Return an existing cell or create a new one."""

        key = (row, col)

        if key not in self.cells:
            self.cells[key] = MapCell()

        return self.cells[key]

    def insert_point(
        self,
        x: float,
        y: float,
        z: float,
        timestamp: float | None = None,
    ) -> MapCell:
        """Insert one LiDAR point into the grid."""

        row, col = self.point_to_index(x, y)

        cell = self.get_or_create_cell(row, col)

        cell.add_observation(
            height=z,
            timestamp=timestamp,
        )

        return cell

    def insert_points(
        self,
        points: np.ndarray,
        timestamp: float | None = None,
    ) -> None:
        """
        Insert an Nx3 array of LiDAR points.
        """

        points = np.asarray(points, dtype=np.float32)

        if points.ndim != 2 or points.shape[1] != 3:
            raise ValueError(
                "points must have shape (N, 3)."
            )

        for x, y, z in points:
            self.insert_point(
                x=x,
                y=y,
                z=z,
                timestamp=timestamp,
            )

    @property
    def num_cells(self) -> int:
        return len(self.cells)
