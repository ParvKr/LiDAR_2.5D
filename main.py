"""Inspect one SemanticKITTI frame and build a foveated 2.5D semantic map."""

import argparse
from pathlib import Path

import numpy as np

from data.datasets.semantic_kitti import SemanticKITTIDataset
from mapping.adaptive_grid import AdaptiveGrid


DRIVABLE = {9, 10, 11, 12, 17}
DYNAMIC = {1, 2, 3, 4, 5, 6, 7, 8}
COLOR_DRIVABLE = (0.39, 0.60, 0.13)
COLOR_STATIC = (0.85, 0.35, 0.19)
COLOR_DYNAMIC = (0.83, 0.33, 0.49)
DEFAULT_BANDS = ((10.0, 0.05), (25.0, 0.10), (50.0, 0.20), (100.0, 0.50))


def classes_to_colors(learning_ids: np.ndarray) -> np.ndarray:
    """Map learning IDs to drivable, static-obstacle, or dynamic colors."""
    colors = np.tile(COLOR_STATIC, (len(learning_ids), 1))
    colors[np.isin(learning_ids, tuple(DRIVABLE))] = COLOR_DRIVABLE
    colors[np.isin(learning_ids, tuple(DYNAMIC))] = COLOR_DYNAMIC
    return colors


def build_grid(points: np.ndarray, labels: np.ndarray) -> AdaptiveGrid:
    """Project labelled points inside 100 m into the foveated 2.5D grid."""
    grid = AdaptiveGrid(list(DEFAULT_BANDS))
    xyz = points[:, :3]
    in_range = np.hypot(xyz[:, 0], xyz[:, 1]) <= grid.max_distance
    grid.insert_points(xyz[in_range], semantic_classes=labels[in_range])
    return grid


def parse_args() -> argparse.Namespace:
    project_root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sequence-dir", type=Path,
                        default=project_root / "SemanticKITTI" / "sequences" / "00")
    parser.add_argument("--frame", type=int, default=0)
    parser.add_argument("--no-visualize", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    dataset = SemanticKITTIDataset(args.sequence_dir)
    points, labels = dataset[args.frame]
    grid = build_grid(points, labels)
    print(f"Loaded {len(points):,} points; projected into {grid.num_cells:,} foveated cells.")
    print(f"Grid bands: {DEFAULT_BANDS} m/resolution; map range: {grid.max_distance:g} m")
    if args.no_visualize:
        return

    import open3d as o3d
    cloud = o3d.geometry.PointCloud()
    cloud.points = o3d.utility.Vector3dVector(points[:, :3])
    cloud.colors = o3d.utility.Vector3dVector(classes_to_colors(labels))
    axes = o3d.geometry.TriangleMesh.create_coordinate_frame(size=2.0)
    o3d.visualization.draw_geometries([cloud, axes], window_name=f"SemanticKITTI frame {args.frame:06d}")


if __name__ == "__main__":
    main()
