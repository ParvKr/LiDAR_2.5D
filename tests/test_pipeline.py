import numpy as np

from LiDAR_25D.geometry.point_cloud import PointCloud
from LiDAR_25D.mapping.adaptive_grid import AdaptiveGrid


def test_xyzi_to_adaptive_map():
    points_xyzi = np.array([
        [2.0, 0.0, 1.0, 0.5],
        [2.1, 0.1, 1.1, 0.6],
        [20.0, 0.0, 1.2, 0.7],
        [80.0, 0.0, 1.5, 0.8],
    ], dtype=np.float32)

    point_cloud = PointCloud.from_xyzi(points_xyzi)

    assert point_cloud.num_points == 4
    assert point_cloud.points_sensor.shape == (4, 3)
    assert point_cloud.intensity.shape == (4,)

    grid = AdaptiveGrid([
        (10.0, 0.05),
        (25.0, 0.10),
        (50.0, 0.20),
        (100.0, 0.50),
    ])

    grid.insert_points(
        point_cloud.get_sensor_xyz()
    )

    assert grid.num_cells > 0

    resolutions = {
        grid.resolution_bands[key[2]][1]
        for key in grid.cells
    }

    assert 0.05 in resolutions
    assert 0.10 in resolutions
    assert 0.50 in resolutions
