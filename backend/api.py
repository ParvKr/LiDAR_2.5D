import sys
from pathlib import Path

# Add project root and ai/ to path so we can import our pipeline modules
PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(PROJECT_ROOT / "ai"))

import numpy as np
import torch
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware

from data.datasets.semantic_kitti import SemanticKITTIDataset
from geometry.bev import BEVProjector
from mapping.adaptive_grid import AdaptiveGrid
from models.unet import UNet

app = FastAPI(title="LiDAR 2.5D Pipeline API")

# Allow requests from our Next.js frontend (which usually runs on port 3000)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# We enforce CPU here because PyTorch's MPS (Mac GPU) backend currently has 
# a known bug that causes random hard crashes/segfaults with Atrous Convolutions (ASPP).
device = torch.device("cpu")
model = None
dataset = None
DEFAULT_BANDS = ((10.0, 0.05), (25.0, 0.10), (50.0, 0.20), (100.0, 0.50))

@app.on_event("startup")
def load_resources():
    global model, dataset
    
    print(f"Starting API. Compute device: {device}")
    
    # 1. Load Dataset
    seq_dir = PROJECT_ROOT.parent / "SemanticKITTI" / "sequences" / "00"
    if seq_dir.exists():
        try:
            dataset = SemanticKITTIDataset(seq_dir)
            print(f"Loaded SemanticKITTI sequence with {len(dataset)} frames.")
        except Exception as e:
            print(f"Warning: Failed to load dataset: {e}")
    else:
        print(f"Warning: Dataset not found at {seq_dir}")

    # 2. Load Model
    model = UNet(in_channels=6, num_classes=20).to(device)
    model.eval()
    
    weights_path = PROJECT_ROOT / "best_unet.pth"
    if weights_path.exists():
        model.load_state_dict(torch.load(weights_path, map_location=device))
        print(f"Successfully loaded trained weights from {weights_path}")
    else:
        print("Warning: best_unet.pth not found. The API will use untrained weights for now.")

@app.get("/api/frame/{frame_id}")
def get_frame(frame_id: int):
    global model, dataset
    
    if dataset is None or frame_id < 0 or frame_id >= len(dataset):
        raise HTTPException(status_code=404, detail="Frame not found or dataset missing.")

    # 1. Load Data
    points, labels = dataset[frame_id]
    
    # 2. BEV Projection & Inference
    projector = BEVProjector()
    bev_result = projector.project(points, labels)
    features_tensor = torch.from_numpy(bev_result.features).float().unsqueeze(0).to(device)
    
    with torch.no_grad():
        logits = model(features_tensor)
        preds = logits.argmax(dim=1).squeeze(0).cpu().numpy()
        
    # 3. Map Back to 3D
    flat_preds = preds.flatten()
    # Initialize with 255 (UNSEEN) so we don't accidentally treat points outside our BEV as class 0 (UNKNOWN)
    point_preds = np.full(len(points), 255, dtype=np.int32)
    valid_points = bev_result.point_to_cell >= 0
    point_preds[valid_points] = flat_preds[bev_result.point_to_cell[valid_points]]
    
    # 4. Adaptive Grid Compression
    grid = AdaptiveGrid(list(DEFAULT_BANDS))
    xyz = points[:, :3]
    in_range = np.hypot(xyz[:, 0], xyz[:, 1]) <= grid.max_distance
    grid.insert_points(xyz[in_range], semantic_classes=point_preds[in_range])
    
    # 5. Extract cells for frontend rendering
    # We send flat arrays to keep the JSON payload small and fast to parse
    positions = []
    colors = []
    sizes = []
    collision_warning = False
    
    for (row, col, band_index), cell in grid.cells.items():
        res = grid.resolution_bands[band_index][1]
        x = (col + 0.5) * res
        y = (row + 0.5) * res
        z = cell.height if cell.height is not None else 0.0
        
        # --- COLLISION AVOIDANCE SYSTEM ---
        # Dynamic objects are classes 1 through 8.
        # If they are within 7 meters, trigger an alarm!
        if cell.semantic_class in {1, 2, 3, 4, 5, 6, 7, 8}:
            if np.hypot(x, y) < 7.0:
                collision_warning = True
                
        positions.extend([x, y, z])
        colors.append(cell.semantic_class if cell.semantic_class is not None else 0)
        sizes.append(res)
        
    reduction_factor = len(points) / max(1, grid.num_cells)
    
    return {
        "frame_id": frame_id,
        "metrics": {
            "raw_points_count": len(points),
            "foveated_cells_count": grid.num_cells,
            "memory_reduction_factor": round(reduction_factor, 1),
            "collision_warning": collision_warning
        },
        "grid": {
            "positions": positions,  # [x1, y1, z1, x2, y2, z2...]
            "labels": colors,        # [class1, class2...]
            "sizes": sizes           # [res1, res2...]
        }
    }
