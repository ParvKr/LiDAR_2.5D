from pathlib import Path

import numpy as np

from data.datasets.semantic_kitti import SemanticKITTIDataset
from perception.taxonomy import SemanticClass


# CHANGE THIS
SEQUENCE_PATH = Path(
    "../SemanticKITTI/sequences/00"
)


def main():
    dataset = SemanticKITTIDataset(SEQUENCE_PATH)

    print(f"Number of frames: {len(dataset)}")

    # Load first real frame
    points, labels = dataset[0]

    print(f"\nFrame 000000")
    print(f"Points: {len(points)}")
    print(f"Point shape: {points.shape}")
    print(f"Point dtype: {points.dtype}")

    # XYZ
    xyz = points[:, :3]
    intensity = points[:, 3]

    print("\nXYZ ranges:")
    print(f"  X: {xyz[:, 0].min():.2f} -> {xyz[:, 0].max():.2f}")
    print(f"  Y: {xyz[:, 1].min():.2f} -> {xyz[:, 1].max():.2f}")
    print(f"  Z: {xyz[:, 2].min():.2f} -> {xyz[:, 2].max():.2f}")

    print("\nIntensity:")
    print(f"  Min: {intensity.min():.2f}")
    print(f"  Max: {intensity.max():.2f}")

    # Horizontal distance from LiDAR
    distance = np.hypot(
        xyz[:, 0],
        xyz[:, 1],
    )

    print("\nHorizontal distance:")
    for limit in [10, 25, 50, 100]:
        count = np.sum(distance <= limit)
        print(f"  <= {limit:3} m: {count}")

    print("\nLabels:")
    unique_labels, counts = np.unique(
        labels,
        return_counts=True,
    )

    for label, count in zip(unique_labels, counts):
        class_name = SemanticClass(label).name
        print(
            f"  {label:2d} "
            f"{class_name:16s}: {count}"
        )

    x = xyz[:, 0]
    y = xyz[:, 1]

    mask = (
            (x >= -50) & (x <= 50) &
            (y >= -50) & (y <= 50)
    )

    print("\nBEV crop:")
    print(f"Points inside ±50m: {np.sum(mask)}")
    print(f"Points outside:      {np.sum(~mask)}")
    print(f"Percentage retained:  {100 * np.mean(mask):.2f}%")


if __name__ == "__main__":
    main()