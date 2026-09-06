# LiDAR 2.5D Foveated Perception Pipeline

This document explains the end-to-end architecture of our autonomous navigation perception pipeline. The goal is to accurately classify 3D LiDAR point clouds and compress them into a memory-efficient "foveated" 2.5D grid in real-time.

---

## 1. Data Ingestion (`data/`)
The pipeline begins by loading raw LiDAR sweeps (point clouds) from the **SemanticKITTI** dataset.
* **Input:** A tensor of shape `(N, 4)` representing $N$ points, where each point has `(x, y, z, intensity)`.
* **Taxonomy:** The raw SemanticKITTI labels are mapped into 20 learning classes, which are further grouped into **Drivable Terrain**, **Static Obstacles** (walls, poles), and **Dynamic Objects** (pedestrians, vehicles).

## 2. 3D to 2.5D BEV Projection (`geometry/`)
Processing millions of 3D points via Sparse 3D Convolutions is computationally expensive. To achieve real-time latency, we project the 3D points into a uniform **Bird's-Eye View (BEV)** grid.
* The z-axis is squashed into 5 descriptive channels: `[max_height, min_height, mean_height, mean_intensity, density]`.
* **Output:** A dense 2D image-like tensor of shape `(5, H, W)`.
* We also maintain a `point_to_cell` mapping array, which remembers exactly which 2D pixel each original 3D point fell into.

## 3. Deep Learning Segmentation (`models/`)
The 5-channel BEV tensor is fed into a custom **UNet Architecture**.
* **Encoder/Decoder:** Standard U-Net pathways preserve high-resolution spatial details needed for small obstacles.
* **ASPP Bottleneck:** To increase the *receptive field*, the bottleneck uses an **Atrous Spatial Pyramid Pooling (ASPP)** module. By running parallel dilated convolutions (rates: 1, 6, 12, 18), the network can look at local details (a person) and global context (the road they are standing on) simultaneously.
* **Regularization:** Dropout (`p=0.2`) is applied in the bottleneck and decoder stages to prevent overfitting.
* **Loss Function:** We train the network using **Focal Loss** rather than standard Cross Entropy. This prevents the network from being overwhelmed by majority classes (like roads) and forces it to focus on rare, critical classes (like bicycles and pedestrians).

## 4. 2D to 3D Re-projection
Once the network outputs a 2D segmentation map `(H, W)`, we use the `point_to_cell` array to instantly assign the predicted 2D classes back to the raw 3D points. 
* **Result:** Every original 3D point is now tagged with a predicted semantic class.

## 5. Foveated Adaptive Grid Mapping (`mapping/`)
Storing the entire environment as a high-resolution 3D voxel grid wastes massive amounts of memory. Instead, we insert the classified 3D points into an **Adaptive 2.5D Grid**.
* **Foveation:** Similar to human vision, resolution is highest near the sensor (where safety is critical) and decreases further away.
  * Distance `< 10m` ➔ `5cm` resolution
  * Distance `< 25m` ➔ `10cm` resolution
  * Distance `< 50m` ➔ `20cm` resolution
  * Distance `< 100m`➔ `50cm` resolution
* **Output:** A sparse dictionary of `MapCell` objects, drastically reducing the number of entities in memory compared to raw point clouds.

## 6. Real-Time Dashboard (`main.py`)
The `main.py` script ties the entire pipeline together into an interactive **Open3D Dashboard**.
* It runs the inference loop.
* It prints the memory reduction factor (e.g., `15x fewer entities`).
* It renders the raw point cloud (colored by the network's predictions) on the bottom, and a visual representation of the foveated grid cells shifted 5 meters above it. This visually proves the adaptive spatial representation to the user.
