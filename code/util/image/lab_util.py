# -----------------------------------------------------------------------------
# Stroke Transfer for Participating Media
# http://cg.it.aoyama.ac.jp/yonghao/sig25/abstsig25.html
#
# File: util/image/lab_util.py
# Maintainer: Hideki Todo
#
# Description:
# Utility functions for color conversion RGB <-> CIE LAB color space.
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
import numpy as np

# Reference white point for D65
REF_WHITE = np.array([0.95047, 1.0, 1.08883])

# RGB to XYZ conversion matrix (sRGB, D65)
M_RGB2XYZ = np.array([
    [0.4124564, 0.3575761, 0.1804375],
    [0.2126729, 0.7151522, 0.0721750],
    [0.0193339, 0.1191920, 0.9503041]
])

# Inverse of the RGB→XYZ matrix
M_XYZ2RGB = np.linalg.inv(M_RGB2XYZ)

# Constants for XYZ↔LAB conversion
EPSILON = 0.008856
KAPPA = 903.3


def rgb_to_xyz(rgb):
    """Convert an RGB image to the XYZ color space.

        Args:
            rgb (np.ndarray): Input image in RGB format, shape (H, W, 3), range [0, 1].

        Returns:
            np.ndarray: Image in XYZ color space, shape (H, W, 3).
        """
    h, w = rgb.shape[:2]
    rgb = rgb.reshape(-1, 3)

    # rgb_linear = np.where(rgb > 0.04045, ((rgb + 0.055) / 1.055) ** 2.4, rgb / 12.92)
    rgb_linear = np.empty_like(rgb)
    mask = rgb > 0.04045
    rgb_linear[mask] = ((rgb[mask] + 0.055) / 1.055) ** 2.4
    rgb_linear[~mask] = rgb[~mask] / 12.92

    xyz = np.dot(rgb_linear, M_RGB2XYZ.T)
    xyz = xyz.reshape(h, w, -1)
    return xyz


def xyz_to_lab(xyz):
    """Convert an XYZ image to the LAB color space.

        Args:
            xyz (np.ndarray): Input image in XYZ format, shape (H, W, 3).

        Returns:
            np.ndarray: Image in LAB color space, shape (H, W, 3).
        """
    h, w = xyz.shape[:2]
    xyz = xyz.reshape(-1, 3)

    xyz_normalized = np.array(xyz)

    for ci in range(3):
        xyz_normalized[:, ci] = xyz[:, ci] / REF_WHITE[ci]

    # mask = xyz_normalized > epsilon
    # xyz_f = np.where(mask, xyz_normalized ** (1 / 3), (kappa * xyz_normalized + 16) / 116)
    xyz_f = np.empty_like(xyz_normalized)
    mask = xyz_normalized > EPSILON
    xyz_f[mask] = xyz_normalized[mask] ** (1 / 3)
    xyz_f[~mask] = (KAPPA * xyz_normalized[~mask] + 16) / 116

    L = 116 * xyz_f[:, 1] - 16
    a = 500 * (xyz_f[:, 0] - xyz_f[:, 1])
    b = 200 * (xyz_f[:, 1] - xyz_f[:, 2])

    lab = np.vstack((L, a, b)).T
    lab = lab.reshape(h, w, -1)
    return lab


def rgb_to_lab(I):
    """Convert an RGB image directly to LAB color space.

        Args:
            I (np.ndarray): Input image in RGB format, shape (H, W, 3), range [0, 1].

        Returns:
            np.ndarray: Image in LAB format, shape (H, W, 3).
        """
    xyz = rgb_to_xyz(I)
    lab = xyz_to_lab(xyz)

    return lab


def lab_to_xyz(lab):
    """Convert a LAB image to the XYZ color space.

        Args:
            lab (np.ndarray): Input image in LAB format, shape (H, W, 3).

        Returns:
            np.ndarray: Image in XYZ format, shape (H, W, 3).
        """
    h, w = lab.shape[:2]
    lab = lab.reshape(-1, 3)

    fy = (lab[:, 0] + 16) / 116
    fz = fy - lab[:, 2] / 200
    fx = lab[:, 1] / 500 + fy

    fxz = np.vstack((fx, fy, fz)).T

    # mask = fxz ** 3 > epsilon
    # xyz = np.where(mask, (fxz ** 3), (116 * fxz - 16) / kappa)
    xyz = np.empty_like(fxz)
    mask = (fxz ** 3) > EPSILON
    xyz[mask] = fxz[mask] ** 3
    xyz[~mask] = (116 * fxz[~mask] - 16) / KAPPA

    for ci in range(3):
        xyz[:, ci] *= REF_WHITE[ci]
    xyz = xyz.reshape(h, w, -1)
    return xyz


def xyz_to_rgb(xyz):
    """Convert an XYZ image to RGB color space.

        Args:
            xyz (np.ndarray): Input image in XYZ format, shape (H, W, 3).

        Returns:
            np.ndarray: Output image in RGB format, shape (H, W, 3), range [0, 1].
        """
    h, w = xyz.shape[:2]
    xyz = xyz.reshape(-1, 3)

    rgb_linear = np.dot(xyz, M_XYZ2RGB.T)
    rgb_linear = np.clip(rgb_linear, 0.0, 5.0)

    # rgb = np.where(rgb_linear > 0.0031308, 1.055 * (rgb_linear ** (1 / 2.4)) - 0.055, 12.92 * rgb_linear)
    rgb = np.empty_like(rgb_linear)
    mask = rgb_linear > 0.0031308
    rgb[mask] = 1.055 * (rgb_linear[mask] ** (1 / 2.4)) - 0.055
    rgb[~mask] = 12.92 * rgb_linear[~mask]

    rgb = rgb.reshape(h, w, -1)
    return rgb


def lab_to_rgb(lab):
    """Convert a LAB image to RGB color space.

        Args:
            lab (np.ndarray): Input image in LAB format, shape (H, W, 3).

        Returns:
            np.ndarray: Output image in RGB format, shape (H, W, 3), range [0, 1].
        """
    h, w = lab.shape[:2]
    xyz = lab_to_xyz(lab.reshape(-1, 3))
    rgb = xyz_to_rgb(xyz)
    return rgb.reshape(h, w, -1)
