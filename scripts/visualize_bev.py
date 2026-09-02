from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from data.datasets.semantic_kitti import SemanticKITTIDataset
from geometry.bev import BEVProjector
from perception.taxonomy import SemanticClass


SEQUENCE_PATH = Path(
    "../SemanticKITTI/sequences/00"
)


def main():
    dataset = SemanticKITTIDataset(SEQUENCE_PATH)

    points, labels = dataset[0]

    projector = BEVProjector()

    result = projector.project(
        points,
        labels,
    )

    height = result.features[0]
    intensity = result.features[3]
    density = result.features[4]

    semantic = result.labels

    extent = [
        projector.x_min,
        projector.x_max,
        projector.y_min,
        projector.y_max,
    ]

    fig, axes = plt.subplots(
        2,
        2,
        figsize=(12, 12),
    )

    axes[0, 0].imshow(
        height,
        origin="lower",
        extent=extent,
    )
    axes[0, 0].set_title("Max Height")
    axes[0, 0].set_xlabel("X (m)")
    axes[0, 0].set_ylabel("Y (m)")

    axes[0, 1].imshow(
        intensity,
        origin="lower",
        extent=extent,
    )
    axes[0, 1].set_title("Mean Intensity")
    axes[0, 1].set_xlabel("X (m)")
    axes[0, 1].set_ylabel("Y (m)")

    axes[1, 0].imshow(
        density,
        origin="lower",
        extent=extent,
    )
    axes[1, 0].set_title("Log Point Density")
    axes[1, 0].set_xlabel("X (m)")
    axes[1, 0].set_ylabel("Y (m)")

    axes[1, 1].imshow(
        semantic,
        origin="lower",
        interpolation="nearest",
    )
    axes[1, 1].set_title("Semantic Ground Truth")
    axes[1, 1].set_xlabel("Column")
    axes[1, 1].set_ylabel("Row")

    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()