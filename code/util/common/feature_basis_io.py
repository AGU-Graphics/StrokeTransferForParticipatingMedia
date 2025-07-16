# -----------------------------------------------------------------------------
# Stroke Transfer for Participating Media
# http://cg.it.aoyama.ac.jp/yonghao/sig25/abstsig25.html
#
# File: util/common/feature_basis_io.py
# Maintainer: Hideki Todo
#
# Description:
# Feature/Basis I/O for PNG and HDF5.
# Note: RGBA images flipped for Y-up.
#
# This file is part of the Stroke Transfer for Participating Media project.
# Released under the Creative Commons Attribution-NonCommercial (CC-BY-NC) license.
# See https://creativecommons.org/licenses/by-nc/4.0/ for details.
#
# DISCLAIMER:
# This code is provided "as is", without warranty of any kind, express or implied,
# including but not limited to the warranties of merchantability, fitness for a
# particular purpose, and noninfringement. In no event shall the authors or
# copyright holders be liable for any claim, damages or other liability.
# -----------------------------------------------------------------------------
import os

import cv2
import h5py
import numpy as np


def load_hdf5(file_path):
    """
    Load feature or basis data from an HDF5 file with resolution metadata.

    Feature data is typically stored as (H, W), and basis data as (H, W, C).

    Args:
        file_path: Path to the HDF5 (.h5 or .hdf5) file.

    Returns:
        A numpy array of shape (H, W) or (H, W, C), depending on content.

    Raises:
        FileNotFoundError: If the file does not exist.
    """
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"File not found: {file_path}")

    with h5py.File(file_path, "r") as f:
        resolution = np.array(f["resolution"][()]).flatten()
        width, height = resolution[:2]
        data = np.array(f["data"][()])

        if data.size == width * height:
            data = data.reshape((height, width))
        else:
            data = data.reshape((height, width, -1))
    return data


def save_hdf5(file_path, data):
    """
    Save feature or basis data to an HDF5 file with resolution metadata.

    Feature data is typically (H, W), and basis data is (H, W, C).

    Args:
        file_path: Destination path for the HDF5 file.
        data: Numpy array of shape (H, W) or (H, W, C) to save.

    Raises:
        FileNotFoundError: If saving fails or file was not created.
    """
    os.makedirs(os.path.dirname(file_path), exist_ok=True)

    height, width = data.shape[:2]
    resolution = np.array([width, height])

    with h5py.File(file_path, "w") as f:
        f.create_dataset("resolution", data=resolution, compression="gzip")
        f.create_dataset("data", data=data, compression="gzip")

    if not os.path.exists(file_path):
        raise IOError(f"Failed to save: {file_path}")


def load_rgba(file_path):
    """
    Load an RGBA image and convert it to float32 in [0, 1].
    Note: Flipping for Y-up is handled in load_image().

    Args:
        file_path: Path to a 4-channel PNG image.

    Returns:
        A numpy array of shape (H, W, 4), dtype float32, or None if file not found.
    """
    if not os.path.exists(file_path):
        return None

    image = cv2.imread(file_path, cv2.IMREAD_UNCHANGED)
    if image is None or image.shape[-1] != 4:
        raise ValueError(f"Image must have 4 channels: {file_path}")

    image = np.float32(image) / 255.0
    image = cv2.cvtColor(image, cv2.COLOR_BGRA2RGBA)
    return image


def save_rgba(file_path, image):
    """
    Save an RGBA image as a PNG file in 8-bit format.
    Note: Flipping for Y-up is handled in save_image().

    Args:
        file_path: Destination path for the output PNG image.
        image: Numpy array of shape (H, W, 4), values in [0, 1] or uint8.
    """
    image_8u = np.uint8(np.clip(image * 255, 0, 255))  # Clamp to [0, 255]
    image_8u = cv2.cvtColor(image_8u, cv2.COLOR_RGBA2BGRA)
    os.makedirs(os.path.dirname(file_path), exist_ok=True)
    success = cv2.imwrite(file_path, image_8u)

    if not success:
        raise IOError(f"Failed to save RGBA image: {file_path}")


def load_image(file_path):
    """
    Load an image from file, supporting both PNG and HDF5 formats.

    Args:
        file_path: Input path ending with .png, .h5, or .hdf5.

    Returns:
        Numpy array of shape (H, W) or (H, W, C), or None if the file does not exist.
    """
    if not os.path.exists(file_path):
        return None

    ext = os.path.splitext(file_path)[1].lower()
    if ext == ".png":
        return np.flipud(load_rgba(file_path))
    elif ext in {".h5", ".hdf5"}:
        return load_hdf5(file_path)
    else:
        raise ValueError(f"Unsupported file extension: {file_path}")


def save_image(file_path, image):
    """
    Save an image to file, either as PNG or HDF5 depending on extension.

    Args:
        file_path: Output path ending with .png, .h5, or .hdf5.
        image: Numpy array of shape (H, W) or (H, W, C) to save.

    Raises:
        ValueError: If the file extension is not supported.
    """
    ext = os.path.splitext(file_path)[1].lower()
    if ext == ".png":
        save_rgba(file_path, np.flipud(image))
    elif ext in {".h5", ".hdf5"}:
        save_hdf5(file_path, image)
    else:
        raise ValueError(f"Unsupported file extension: {file_path}")
