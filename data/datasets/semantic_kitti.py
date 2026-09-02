from pathlib import Path
from geometry.point_cloud import PointCloud

import numpy as np


class SemanticKITTIDataset:
    """
    Minimal loader for a SemanticKITTI-style sequence.

    Each LiDAR frame contains Nx4 values:
        x, y, z, intensity

    Each labels file contains one labels per point.
    """

    def __init__(
        self,
        sequence_dir: str | Path,
    ):
        self.sequence_dir = Path(sequence_dir)

        self.velodyne_dir = self.sequence_dir / "velodyne"
        self.labels_dir = self.sequence_dir / "labels"

        if not self.velodyne_dir.exists():
            raise FileNotFoundError(
                f"Velodyne directory not found: "
                f"{self.velodyne_dir}"
            )

        self.point_files = sorted(
            path for pattern in ("*.velodyne", "*.bin")
            for path in self.velodyne_dir.glob(pattern)
        )

        if not self.point_files:
            raise FileNotFoundError(
                f"No .velodyne or .bin point-cloud files found in "
                f"{self.velodyne_dir}"
            )

    def __len__(self) -> int:
        return len(self.point_files)

    def load_points(self, index: int) -> np.ndarray:
        """
        Load one LiDAR frame.

        Returns:
            Nx4 float32 array:
            [x, y, z, intensity]
        """

        path = self.point_files[index]

        points = np.fromfile(
            path,
            dtype=np.float32,
        )

        if points.size % 4 != 0:
            raise ValueError(
                f"Invalid point cloud file: {path}"
            )

        return points.reshape(-1, 4)

    def load_point_cloud(
            self,
            index: int,
            timestamp: float | None = None,
    ) -> PointCloud:
        """
        Load one SemanticKITTI frame as a PointCloud object.
        """

        points_xyzi = self.load_points(index)

        return PointCloud.from_xyzi(
            points_xyzi,
            timestamp=timestamp,
        )

    def load_raw_labels(self, index: int) -> np.ndarray:
        """
        Load raw 32-bit SemanticKITTI labels.

        The raw values contain both semantic and instance
        information.
        """

        frame_name = self.point_files[index].stem
        label_path = self.labels_dir / f"{frame_name}.labels"
        if not label_path.exists():
            label_path = self.labels_dir / f"{frame_name}.label"

        if not label_path.exists():
            raise FileNotFoundError(
                f"Label file not found: {label_path}"
            )

        return np.fromfile(
            label_path,
            dtype=np.uint32,
        )

    def load_labels(self, index: int) -> np.ndarray:
        """
        Load labels converted to SemanticKITTI learning IDs.
        """

        from perception.taxonomy import map_semantic_kitti_label

        raw_labels = self.load_raw_labels(index)

        return np.array(
            [
                int(map_semantic_kitti_label(label))
                for label in raw_labels
            ],
            dtype=np.int32,
        )

    def __getitem__(self, index: int) -> tuple[np.ndarray, np.ndarray]:
        """
        Return points and labels for one frame.
        """

        points = self.load_points(index)
        labels = self.load_labels(index)

        if len(points) != len(labels):
            raise ValueError(
                "Number of points and labels must match."
            )

        return points, labels
