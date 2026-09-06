# LiDAR_25D

A 2.5D LiDAR perception and mapping project based on the SemanticKITTI dataset, fully equipped with a modern Web Dashboard.

## Architecture

This project is a full-stack monorepo divided into three main components:
- **`ai/`**: The Core Python Machine Learning pipeline (Data loading, 3D->2.5D BEV projection, UNet segmentation, Adaptive Grid Mapping).
- **`backend/`**: A FastAPI server that keeps the PyTorch model in memory and streams compressed 2.5D foveated grids.
- **`frontend/`**: A Vite + React application that renders the 3D dashboard using `React-Three-Fiber` and `InstancedMesh`.

## Setup

It is recommended to use a virtual environment for the AI and Backend:

```bash
python -m venv venv
source venv/bin/activate
pip install -e .[dev,web]
```

For the frontend:
```bash
cd frontend
npm install
```

## Dataset (SemanticKITTI)

Place the SemanticKITTI dataset at the root of the project. Ensure the folder structure follows:
```
SemanticKITTI/
  sequences/
    00/
      velodyne/
      labels/
```

## Usage

### 1. Training the AI
To train the UNet segmentation model on SemanticKITTI:
```bash
python ai/scripts/train_single.py --epochs 20 --learning-rate 1e-3
```
This will save a `best_unet.pth` file in the root directory.

### 2. Running the Dashboard
You need two terminals to run the full-stack web application.

**Terminal 1 (Backend):**
```bash
python -m uvicorn backend.api:app --reload
```

**Terminal 2 (Frontend):**
```bash
cd frontend
npm run dev
```
Open the provided `localhost` link in your browser to view the real-time 3D dashboard!

## Contributing

We use `ruff` for formatting and linting, and `mypy` for static type checking. 

Run the tests before submitting changes:
```bash
pytest ai/tests/
ruff check ai/ backend/
mypy ai/ backend/
```
