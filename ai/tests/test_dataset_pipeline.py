import numpy as np

from data.datasets.semantic_kitti import SemanticKITTIDataset
from mapping.adaptive_grid import AdaptiveGrid
from perception.taxonomy import SemanticClass


def test_dataset_to_adaptive_grid(tmp_path):
    sequence_dir = tmp_path / "00"

    velodyne_dir = sequence_dir / "velodyne"
    labels_dir = sequence_dir / "labels"

    velodyne_dir.mkdir(parents=True)
    labels_dir.mkdir(parents=True)

    points = np.array([
        [2.0, 0.0, 1.0, 0.5],
        [2.1, 0.1, 1.1, 0.6],
        [20.0, 0.0, 1.2, 0.7],
        [80.0, 0.0, 1.5, 0.8],
    ], dtype=np.float32)

    points.tofile(
        velodyne_dir / "000000.bin"
    )

    raw_labels = np.array([
        10,
        10,
        18,
        30,
    ], dtype=np.uint32)

    raw_labels.tofile(
        labels_dir / "000000.label"
    )

    dataset = SemanticKITTIDataset(
        sequence_dir
    )

    point_cloud = dataset.load_point_cloud(0)
    labels = dataset.load_labels(0)

    assert point_cloud.num_points == 4
    assert len(labels) == 4

    assert labels[0] == SemanticClass.CAR
    assert labels[2] == SemanticClass.TRUCK
    assert labels[3] == SemanticClass.PERSON

    grid = AdaptiveGrid([
        (10.0, 0.05),
        (25.0, 0.10),
        (50.0, 0.20),
        (100.0, 0.50),
    ])

    grid.insert_points(point_cloud.points_sensor, semantic_classes=labels)

    assert grid.num_cells > 0
    assert any(cell.semantic_class == SemanticClass.CAR for cell in grid.cells.values())
