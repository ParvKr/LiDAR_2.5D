import numpy as np

from LiDAR_25D.geometry.coordinates import transform_points
from LiDAR_25D.geometry.point_cloud import PointCloud


def test_point_cloud():
    points = np.array([
        [1.0, 2.0, 3.0],
        [4.0, 5.0, 6.0],
    ])

    cloud = PointCloud(points)

    assert cloud.num_points == 2
    assert cloud.points_sensor.shape == (2, 3)


def test_translation():
    points = np.array([
        [1.0, 2.0, 3.0],
    ])

    rotation = np.eye(3)
    translation = np.array([10.0, 20.0, 30.0])

    result = transform_points(
        points,
        rotation,
        translation,
    )

    expected = np.array([
        [11.0, 22.0, 33.0],
    ])

    np.testing.assert_allclose(result, expected)