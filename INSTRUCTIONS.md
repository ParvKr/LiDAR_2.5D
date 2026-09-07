# LiDAR 2.5D Pipeline: Complete Setup & Training Guide

This guide covers everything needed to set up the project, process the raw SemanticKITTI data, train the UNet model on a local NVIDIA GPU (like an RTX 3050), and launch the 3D dashboard.

---

## 1. Prerequisites
- **Python 3.10+**
- **NVIDIA GPU** with updated drivers (for CUDA)
- **Node.js** (for running the React frontend)

---

## 2. Environment Setup

Open a terminal and navigate to the root of the project:

```bash
# 1. Create a virtual environment
python -m venv venv

# 2. Activate it
# On Windows:
venv\Scripts\activate
# On Mac/Linux:
source venv/bin/activate

# 3. Install PyTorch with CUDA support (adjust cu118/cu121 based on your drivers)
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118

# 4. Install the rest of the project dependencies
pip install fastapi uvicorn numpy scipy tqdm open3d
```

---

## 3. Dataset Setup

Download the SemanticKITTI dataset and ensure it is organized exactly like this:
```text
SemanticKITTI_Folder/
└── dataset/
    └── sequences/
        ├── 00/
        │   ├── velodyne/
        │   └── labels/
        ├── 01/
        ├── 02/
        ...
```

---

## 4. BEV Cache Generation (Data Preprocessing)

To bypass massive CPU bottlenecks during training, we must pre-generate compressed Bird's Eye View (BEV) arrays of the point clouds. This script uses fully vectorized C-math to aggressively project the LiDAR data into compressed `float16` and `int8` files.

Run this command, replacing the paths with the actual locations on your machine.
*(We recommend setting the `--local-cache-root` to your fastest SSD for maximum speed).*

```bash
python ai/scripts/generate_cache_seq.py \
    --seqs 00 01 02 03 04 05 06 07 08 09 10 \
    --dataset-root "D:\path\to\SemanticKITTI_Folder\dataset" \
    --local-cache-root "C:\temp_bev_cache" \
    --batch-size 100
```
*Note: This process will display progress bars. Wait for all sequences to reach `STATUS: COMPLETE`.*

---

## 5. Model Training

Once the cache is securely stored, you can begin training. The dataset loader will automatically detect the cache, allowing your GPU to hit 100% utilization instantly. 

```bash
python ai/scripts/train_single.py \
    --dataset-root "D:\path\to\SemanticKITTI_Folder\dataset" \
    --checkpoint-dir "./checkpoints" \
    --epochs 50 \
    --patience 10
```

**What to expect:**
- The script uses **Mixed Precision (AMP)** and a larger batch size to maximize RTX 3050 hardware.
- It will train on sequences `00-07, 09` and validate on `08, 10`.
- If the validation loss stops improving for 10 epochs (patience), it will cleanly trigger Early Stopping.
- The best weights will be safely saved to `./checkpoints/best_unet.pth`.

---

## 6. Running the Demo (API + 3D UI)

Once you have your trained weights, copy `./checkpoints/best_unet.pth` and place it directly into the project root folder.

### Step 1: Start the AI Backend
In your activated Python terminal, start the FastAPI server:
```bash
uvicorn backend.api:app --reload --port 8000
```

### Step 2: Start the 3D Dashboard
Open a **new terminal**, navigate to the `frontend/` folder, and start the React app:
```bash
cd frontend
npm install
npm run dev
```

Open your browser to the provided `localhost` link (usually `http://localhost:5173`) to view the real-time Foveated 3D projection, click the **Auto-Drive** button, and watch the Collision Warning system in action!
