"""Inspect one SemanticKITTI frame, run UNet inference, and build a foveated 2.5D semantic map."""

import argparse
import logging
from pathlib import Path
import sys

# Add the 'ai' directory to Python path so internal imports work
sys.path.append(str(Path(__file__).resolve().parents[0]))

import numpy as np
import torch
import open3d as o3d

from data.datasets.semantic_kitti import SemanticKITTIDataset
from geometry.bev import BEVProjector
from mapping.adaptive_grid import AdaptiveGrid
from models.unet import UNet

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

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


def parse_args() -> argparse.Namespace:
    project_root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description="End-to-End Inference Dashboard")
    parser.add_argument(
        "--sequence-dir", 
        type=Path,
        default=project_root.parent / "SemanticKITTI" / "sequences" / "00"
    )
    parser.add_argument("--frame", type=int, default=0)
    parser.add_argument("--no-visualize", action="store_true")
    parser.add_argument("--model-weights", type=Path, help="Path to trained UNet weights (optional)")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Using device: {device}")

    # 1. Initialize Network
    logger.info("Initializing UNet model...")
    model = UNet(in_channels=6, num_classes=20).to(device)
    model.eval()
    
    if args.model_weights and args.model_weights.exists():
        model.load_state_dict(torch.load(args.model_weights, map_location=device))
        logger.info(f"Loaded weights from {args.model_weights}")
    else:
        logger.warning("No weights provided. Model will predict randomly for demonstration.")

    # 2. Load Point Cloud
    logger.info(f"Loading SemanticKITTI sequence frame {args.frame}...")
    dataset = SemanticKITTIDataset(args.sequence_dir)
    points, labels = dataset[args.frame]  # We load labels just for comparison/metrics
    
    # 3. Project to BEV for Neural Network
    logger.info("Projecting to uniform BEV...")
    projector = BEVProjector()
    bev_result = projector.project(points, labels)
    features_tensor = torch.from_numpy(bev_result.features).float().unsqueeze(0).to(device)
    
    # 4. Neural Network Inference
    logger.info("Running UNet inference...")
    with torch.no_grad():
        logits = model(features_tensor)
        preds = logits.argmax(dim=1).squeeze(0).cpu().numpy()  # (H, W)
    
    # 5. Map 2D predictions back to 3D points
    logger.info("Mapping predictions back to 3D points...")
    flat_preds = preds.flatten()
    point_preds = np.zeros(len(points), dtype=np.int32)
    valid_points = bev_result.point_to_cell >= 0
    point_preds[valid_points] = flat_preds[bev_result.point_to_cell[valid_points]]
    
    # 6. Insert classified points into the Foveated Adaptive Grid
    logger.info("Building adaptive foveated 2.5D grid...")
    grid = AdaptiveGrid(list(DEFAULT_BANDS))
    xyz = points[:, :3]
    # after
    in_range = valid_points & (np.hypot(xyz[:, 0], xyz[:, 1]) <= grid.max_distance)
    
    # Insert points with their *PREDICTED* classes
    grid.insert_points(xyz[in_range], semantic_classes=point_preds[in_range])
    
    logger.info(f"Loaded {len(points):,} raw points.")
    logger.info(f"Compressed into {grid.num_cells:,} foveated cells.")
    logger.info(f"Grid bands: {DEFAULT_BANDS} m/resolution")
    logger.info(f"Memory reduction: ~{len(points) / max(1, grid.num_cells):.1f}x fewer entities.")

    if args.no_visualize:
        return

    # 7. Visualization Dashboard
    logger.info("Launching visualization dashboard...")
    
    # Create point cloud visualization for the raw points (colored by predictions)
    cloud = o3d.geometry.PointCloud()
    cloud.points = o3d.utility.Vector3dVector(xyz[in_range])
    cloud.colors = o3d.utility.Vector3dVector(classes_to_colors(point_preds[in_range]))
    
    # Create cell centers representation to show the foveated grid structure
    cell_points = []
    cell_colors = []
    
    for (row, col, band_index), cell in grid.cells.items():
        res = grid.resolution_bands[band_index][1]
        x = (col + 0.5) * res
        y = (row + 0.5) * res
        z = cell.height if cell.height is not None else 0.0
        cell_points.append([x, y, z])
        
        # Color by predicted semantic class
        pred_class = cell.semantic_class if cell.semantic_class is not None else 0
        cell_colors.append(pred_class)
        
    cell_cloud = o3d.geometry.PointCloud()
    if cell_points:
        cell_cloud.points = o3d.utility.Vector3dVector(np.array(cell_points))
        cell_cloud.colors = o3d.utility.Vector3dVector(classes_to_colors(np.array(cell_colors)))
        # Shift the cell representation slightly up to avoid z-fighting with the raw point cloud
        cell_cloud.translate((0, 0, 5.0))
        
    axes = o3d.geometry.TriangleMesh.create_coordinate_frame(size=2.0)
    
    logger.info("Visualizing: Raw points (bottom) and Foveated Grid Cells (top, shifted up 5m)")
    o3d.visualization.draw_geometries([cloud, cell_cloud, axes], window_name=f"Inference Dashboard (Frame {args.frame:06d})")


if __name__ == "__main__":
    main()
