from dataclasses import dataclass
from typing import Optional

import numpy as np

@dataclass
class PointCloud:
    """
    Represents a single LIDAR frame

    points_sensor: Nx3 array containing x, y, z coordinates in the LiDAR coordinate system.
    intensity: Optional N-element array containing return intensity.
    timestamp: Timestamp associated with this frame.
    points_world: Optional Nx3 array containing the same points transformed into a world/reference coordinate system.
    """

    points_sensor: np.ndarray
    intensity: Optional[np.ndarray] = None
    timestamp: Optional[float] = None
    points_world: Optional[np.ndarray] = None

    def __post_init__(self):
        if self.points_sensor.ndim != 2:
            raise ValueError("points_sensor must have shape (N, 3).")

        if self.points_sensor.shape[1] != 3:
            raise ValueError("points_sensor must contain x, y, z coordinates.")

        if self.intensity is not None:
            if len(self.intensity) != len(self.points_sensor):
                raise ValueError(
                    "Intensity must contain one value per point."
                )

        if self.points_world is not None:
            if self.points_world.shape != self.points_sensor.shape:
                raise ValueError(
                    "points_world must have the same shape as points_sensor."
                )

    @classmethod
    def from_xyzi(
            cls,
            points_xyzi: np.ndarray,
            timestamp: Optional[float] = None,
    ) -> "PointCloud":
        """
        Construct a PointCloud from an Nx4 [x, y, z, intensity]
        array.
        """

        points_xyzi = np.asarray(
            points_xyzi,
            dtype=np.float32,
        )

        if points_xyzi.ndim != 2 or points_xyzi.shape[1] != 4:
            raise ValueError(
                "points_xyzi must have shape (N, 4)."
            )

        return cls(
            points_sensor=points_xyzi[:, :3],
            intensity=points_xyzi[:, 3],
            timestamp=timestamp,
        )

    @property
    def num_points(self) -> int:
        return len(self.points_sensor)

    def get_sensor_xyz(self) -> np.ndarray:
        return self.points_sensor

    def get_world_xyz(self) -> np.ndarray:
        if self.points_world is None:
            raise ValueError("World coordinates are not available.")

        return self.points_world