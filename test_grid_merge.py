import numpy as np
from mapping.adaptive_grid import AdaptiveGrid
import time

grid = AdaptiveGrid([(50.0, 0.2)])

# Frame 1: 10 ROAD points in cell (0,0)
pts1 = np.zeros((10, 3))
pts1[:, 0] = 0.1
pts1[:, 1] = 0.1
classes1 = np.full(10, 9)

# Frame 2: 2 CAR points in cell (0,0)
pts2 = np.zeros((2, 3))
pts2[:, 0] = 0.1
pts2[:, 1] = 0.1
classes2 = np.full(2, 1)

grid.insert_points(pts1, classes1)
cell = grid.get_or_create_cell(0, 0, 0)
print(f"After Frame 1: Class {cell.semantic_class}, Conf {cell.semantic_confidence:.3f}")

grid.insert_points(pts2, classes2)
cell = grid.get_or_create_cell(0, 0, 0)
print(f"After Frame 2: Class {cell.semantic_class}, Conf {cell.semantic_confidence:.3f}")

