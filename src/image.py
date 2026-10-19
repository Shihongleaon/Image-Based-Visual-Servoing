"""
Image processing utilities for the IBVS project.
"""

import numpy as np
from PIL import Image


def get_image_config() -> dict:
    """
    returns the image configuration
    """
    return {
        "width": 800,
        "height": 500,
        "fov": 90,
        "aspect": 1.6,
        "near_val": 0.01,
        "far_val": 10,
    }


def convert_img_to_arr(rgb_img, depth_img, seg_img, width, height):
    """
    Convert PyBullet's RGB pixels (flat list) into an RGBA numpy array.
    Handles both RGB (3-channel) and RGBA (4-channel) input.
    """
    arr = np.array(rgb_img, dtype=np.uint8)
    total_pixels = width * height

    # Determine number of channels
    if arr.size == total_pixels * 3:
        # RGB -> reshape and add alpha channel (fully opaque)
        arr = arr.reshape(height, width, 3)
        rgba = np.zeros((height, width, 4), dtype=np.uint8)
        rgba[:, :, :3] = arr
        rgba[:, :, 3] = 255
        return rgba
    elif arr.size == total_pixels * 4:
        # RGBA -> reshape directly
        arr = arr.reshape(height, width, 4)
        return arr
    else:
        raise ValueError(
            f"Unexpected array size: {arr.size}, expected {total_pixels*3} or {total_pixels*4}"
        )


def save_image(error_mag, i, img_arr, min_error):
    """
    Saves the image with the error magnitude and iteration number.
    """
    if error_mag is None:
        filename = f"img/no_marker_step_{i:03d}.png"
    else:
        filename = f"img/step_{i:03d}_err_{error_mag:.4f}.png"
    img = Image.fromarray(img_arr, "RGBA")
    img.save(filename)