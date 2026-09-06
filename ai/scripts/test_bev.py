from pathlib import Path

import numpy as np

from data.datasets.semantic_kitti import SemanticKITTIDataset
from geometry.bev import BEVProjector


SEQUENCE_PATH = Path(__file__).resolve().parents[2] / "SemanticKITTI/sequences/00"


def main():
    dataset = SemanticKITTIDataset(SEQUENCE_PATH)

    points, labels = dataset[0]

    projector = BEVProjector()

    result = projector.project(
        points,
        labels,
    )

    print("BEV")
    print(f"Features shape: {result.features.shape}")
    print(f"Labels shape:   {result.labels.shape}")
    print(f"Mask shape:     {result.valid_mask.shape}")

    print(
        f"Label mask shape: "
        f"{result.label_mask.shape}"
    )

    print(
        f"Trainable cells: "
        f"{result.label_mask.sum()}"
    )

    print(
        f"Occupied cells: "
        f"{result.valid_mask.sum()}"
    )

    print(
        f"Total cells:    "
        f"{result.valid_mask.size}"
    )

    print(
        f"Occupancy:       "
        f"{100 * result.valid_mask.mean():.2f}%"
    )

    print(
        f"Point mappings: "
        f"{np.sum(result.point_to_cell >= 0)}"
    )

    print("\nFeature ranges:")

    names = [
        "max_height",
        "min_height",
        "mean_height",
        "mean_intensity",
        "density",
    ]

    for name, channel in zip(
        names,
        result.features,
    ):
        print(
            f"  {name:16s}: "
            f"{channel.min():.4f} -> "
            f"{channel.max():.4f}"
        )

    occupied_counts = result.features[4]

    nonzero = occupied_counts[occupied_counts > 0]

    counts = result.point_count
    nonzero = counts[counts > 0]

    print("\nPoints per occupied cell:")
    print(f"Min:    {nonzero.min()}")
    print(f"Mean:   {nonzero.mean():.2f}")
    print(f"Median: {np.median(nonzero):.2f}")
    print(f"Max:    {nonzero.max()}")

    counts = result.point_count

    max_index = np.argmax(counts)

    row, col = np.unravel_index(
        max_index,
        counts.shape,
    )

    print("\nMost populated cell:")
    print(f"Row:   {row}")
    print(f"Col:   {col}")
    print(f"Count: {counts[row, col]}")

    x_center = (
            projector.x_min
            + (col + 0.5) * projector.resolution
    )

    y_center = (
            projector.y_min
            + (row + 0.5) * projector.resolution
    )

    print(
        f"Center: ({x_center:.2f}, {y_center:.2f})"
    )

    distance = np.hypot(
        points[:, 0],
        points[:, 1],
    )

    xy_mask = (
            (points[:, 0] >= -50)
            & (points[:, 0] < 50)
            & (points[:, 1] >= -50)
            & (points[:, 1] < 50)
    )

    print("\nPoint filtering:")
    print(f"Inside XY crop: {xy_mask.sum()}")

    for threshold in [0.1, 0.25, 0.5, 1.0, 2.0]:
        mask = xy_mask & (distance >= threshold)

        print(
            f"Distance >= {threshold:4.2f} m: "
            f"{mask.sum()}"
        )

    near_origin = distance < 0.1

    print("\nNear-origin points:")
    print(f"Count: {near_origin.sum()}")

    print("\nXYZ statistics:")
    print(
        points[near_origin, :3].min(axis=0)
    )

    print(
        points[near_origin, :3].max(axis=0)
    )

    print("\nFirst 10:")
    print(points[near_origin][:10])


if __name__ == "__main__":
    main()
