import numpy as np

from LiDAR_25D.mapping.adaptive_grid import AdaptiveGrid
from LiDAR_25D.mapping.uniform_grid import UniformGrid


def test_uniform_grid_accumulates_points():
    grid = UniformGrid(
        resolution=1.0,
        x_min=0.0,
        y_min=0.0,
    )

    points = np.array([
        [0.2, 0.2, 1.0],
        [0.4, 0.3, 2.0],
        [0.7, 0.8, 3.0],
    ])

    grid.insert_points(points)

    assert grid.num_cells == 1

    cell = next(iter(grid.cells.values()))

    assert cell.point_count == 3
    assert cell.height == 2.0
    assert np.isclose(cell.height_variance, 2.0 / 3.0)
    assert cell.occupied


def test_uniform_grid_separates_cells():
    grid = UniformGrid(
        resolution=1.0,
        x_min=0.0,
        y_min=0.0,
    )

    grid.insert_point(0.2, 0.2, 1.0)
    grid.insert_point(1.2, 0.2, 2.0)

    assert grid.num_cells == 2


def test_adaptive_resolution_changes_with_distance():
    grid = AdaptiveGrid([
        (10.0, 0.05),
        (25.0, 0.10),
        (50.0, 0.20),
        (100.0, 0.50),
    ])

    assert grid.get_resolution(5.0) == 0.05
    assert grid.get_resolution(20.0) == 0.10
    assert grid.get_resolution(40.0) == 0.20
    assert grid.get_resolution(80.0) == 0.50


def test_adaptive_grid_retains_semantic_majority_and_rejects_out_of_range():
    grid = AdaptiveGrid([(10.0, 0.05), (100.0, 0.50)])
    grid.insert_points(
        np.array([[2.0, 0.0, 1.0], [2.01, 0.01, 2.0], [2.02, 0.01, 3.0]]),
        semantic_classes=np.array([9, 9, 1]),
    )
    cell = next(iter(grid.cells.values()))
    assert cell.semantic_class == 9
    assert np.isclose(cell.semantic_confidence, 2 / 3)

    with np.testing.assert_raises(ValueError):
        grid.insert_point(101.0, 0.0, 0.0)
