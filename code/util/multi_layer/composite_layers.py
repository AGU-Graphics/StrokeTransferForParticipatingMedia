# -----------------------------------------------------------------------------
# Stroke Transfer for Participating Media
# http://cg.it.aoyama.ac.jp/yonghao/sig25/abstsig25.html
#
# File: util/multi_layer/composite_layers.py
# Maintainer: Hideki Todo
#
# Description:
# Utilities for compositing RGBA mulity-layer images.
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
import os.path

import cv2
import numpy as np
from tqdm import tqdm


def load_rgba(img_file):
    """Loads an RGBA image from file and converts it to float32 in [0, 1] range.

    Args:
        img_file (str): Path to the input image file.

    Returns:
        ndarray: Image array in RGBA format, float32 in [0, 1].
    """
    I = cv2.imread(img_file, -1)
    I = np.float32(I) / 255.0

    I = cv2.cvtColor(I, cv2.COLOR_BGRA2RGBA)
    return I


def save_rgba(out_file, I):
    """Saves a float RGBA image to file after converting to uint8.

    Args:
        out_file (str): Path to the output image file.
        I (ndarray): RGBA image in float32 format, values in [0, 1] or [0, 255].
    """
    I_8u = np.array(I)
    if I_8u.dtype != np.uint8:
        I_8u = np.uint8(255 * I_8u)
    I_8u = cv2.cvtColor(I_8u, cv2.COLOR_RGBA2BGRA)
    cv2.imwrite(out_file, I_8u)


def alpha_blend(I_dst, I_src):
    """Alpha blends a source RGBA image over a destination RGBA image.

    Args:
        I_dst (ndarray): Destination RGBA image.
        I_src (ndarray): Source RGBA image to blend on top.

    Returns:
        ndarray: Blended RGBA image.
    """
    A_src = I_src[:, :, 3]
    A_dst = I_dst[:, :, 3]

    A_out = A_src + (1.0 - A_src) * A_dst

    I_out = np.array(I_dst)

    epsilon = np.finfo(float).eps

    for ci in range(3):
        I_out[:, :, ci] = A_src * I_src[:, :, ci] + (1.0 - A_src) * A_dst * I_dst[:, :, ci]
        I_out[A_out > epsilon, ci] /= A_out[A_out > epsilon]

    I_out[:, :, 3] = A_out
    return I_out


def composite_layers(layer_files, out_file,
                     bg_color=np.array([0.0, 0.0, 0.0, 0.0]),
                     resolution=(768, 768)):
    """Composites multiple RGBA layers and saves the final blended image.

    Args:
        layer_files (list[str]): List of RGBA image file paths to composite.
        out_file (str): Output path for the final composited image.
        bg_color (ndarray): Background RGBA color as float array.
        resolution (tuple): Output resolution (width, height).
    """
    O = np.zeros((resolution[1], resolution[0], 4))
    O[:, :, :] = bg_color

    for layer_file in layer_files:
        if not os.path.exists(layer_file):
            raise FileNotFoundError(f"File not found: {layer_file}")
        I = load_rgba(layer_file)
        I = cv2.resize(I, resolution)
        O = alpha_blend(O, I)

    out_dir = os.path.dirname(out_file)
    os.makedirs(out_dir, exist_ok=True)
    save_rgba(out_file, O)


def composite_layers_frames(
        layer_file_templates, out_image_file_template,
        bg_color=np.array([0.0, 0.0, 0.0, 0.0]),
        resolution=(768, 768),
        frames=range(1, 3)):
    """Composites RGBA image layers over multiple frames and saves them.

    Args:
        layer_file_templates (list[str]): List of filename templates for layer images per frame.
        out_image_file_template (str): Filename template for output images.
        bg_color (ndarray): Background RGBA color.
        resolution (tuple): Output resolution (width, height).
        frames (iterable): List or range of frame indices.
    """
    for frame in tqdm(frames):
        layer_files = [layer_file_template % frame for layer_file_template in layer_file_templates]
        out_file = out_image_file_template % frame

        composite_layers(layer_files, out_file,
                         bg_color=bg_color,
                         resolution=resolution)
