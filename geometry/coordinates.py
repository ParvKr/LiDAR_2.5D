import numpy as np

def transform_points(
    points: np.ndarray,
    rotation: np.ndarray,
    translation: np.ndarray,
) -> np.ndarray:
    """
    Transform Nx3 points using: p_world = R @ p_sensor + t
    Args:
        points: Nx3 point array.
        rotation: 3x3 rotation matrix.
        translation: 3-element translation vector.
    Returns:
        Nx3 transformed points.
    """

    points = np.asarray(points, dtype=np.float32)
    rotation = np.asarray(rotation, dtype=np.float32)
    translation = np.asarray(translation, dtype=np.float32)

    if points.ndim != 2 or points.shape[1] != 3:
        raise ValueError("points must have shape (N, 3).")

    if rotation.shape != (3, 3):
        raise ValueError("rotation must have shape (3, 3).")

    if translation.shape != (3,):
        raise ValueError("translation must have shape (3,).")

    return points @ rotation.T + translation