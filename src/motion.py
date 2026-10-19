"""
Motion control utilities for IBVS.
"""

import numpy as np
from image import get_image_config


def get_error_vec(points: list) -> np.ndarray:
    """
    returns the error vector (difference between current and desired feature positions)
    """
    # desired features (target corners of aruco marker)
    # These are the target pixel coordinates (from target image)
    # You may need to tune these based on your target image
    desired = np.array([
        [380, 228],
        [380, 270],
        [421, 228],
        [421, 270]
    ], dtype=np.float32)
    # current features (only first 3 corners used for 6 DOF)
    current = np.array(points[:3], dtype=np.float32)
    # error = current - desired (using only x,y of first 3 corners)
    error = current.flatten() - desired[:3].flatten()
    return error


def get_error_mag(error: np.ndarray) -> float:
    """
    returns the mean squared error magnitude
    """
    return float(np.mean(error ** 2))


def get_velocity(points: list, depth_buffer: list, width: int = 800, height: int = 500) -> np.ndarray:
    """
    computes the velocity from the current feature points and depth buffer
    """
    LAMBDA = 0.5
    # Stack Jacobian for first 3 points (6 DOF)
    J = []
    for i in range(3):
        x, y = points[i]
        x = int(round(x))
        y = int(round(y))
        # Clamp to image bounds
        x = max(0, min(x, width-1))
        y = max(0, min(y, height-1))
        # Get depth at this pixel (depth_buffer is flat list)
        idx = y * width + x
        Z = depth_buffer[idx]
        if Z == 0:
            Z = 1.0  # avoid division by zero
        # Image Jacobian for point (x,y) with depth Z
        J.append([
            -1/Z, 0, x/Z, x*y, -(1+x*x), y
        ])
        J.append([
            0, -1/Z, y/Z, (1+y*y), -x*y, -x
        ])
    J = np.array(J)
    # Compute pseudo-inverse
    J_pinv = np.linalg.pinv(J)
    error = get_error_vec(points)
    vel = -LAMBDA * np.matmul(J_pinv, error)
    return vel