# -----------------------------------------------------------------------------
# Stroke Transfer for Participating Media
# http://cg.it.aoyama.ac.jp/yonghao/sig25/abstsig25.html
#
# File: util/annotation_interpolation/vis_utils.py
# Maintainer: Hideki Todo
#
# Description:
# Utility functions for visualizing annotation data.
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
import numpy as np
from matplotlib import pyplot as plt


def load_rgba(img_file):
    """
    Load an RGBA image from the given file path and normalize to [0, 1].
    """
    I = cv2.imread(img_file, -1)
    if I.dtype == np.uint16:
        I = np.float32(I) / 65535.0
    else:
        I = np.float32(I) / 255.0
    I = cv2.cvtColor(I, cv2.COLOR_BGRA2RGBA)
    return I


def load_mask(img_file):
    """
    Load an image and extract the mask from the minimum of the red and alpha channels.
    """
    I = cv2.imread(img_file, -1)
    if I.dtype == np.uint16:
        I = np.float32(I) / 65535.0
    else:
        I = np.float32(I) / 255.0
    I = np.minimum(I[:, :, 0], I[:, :, 3])
    return I


def save_fig(file_path, bbox_inches="tight", pad_inches=0.05, transparent=False):
    """
    Save the current matplotlib figure to the given file path.
    """
    fig = plt.gcf()
    os.makedirs(os.path.dirname(file_path), exist_ok=True)
    fig.savefig(file_path, bbox_inches=bbox_inches, pad_inches=pad_inches, transparent=transparent)


def plot_image(image):
    """
    Display an image with white background and axes turned off.
    """
    ax = plt.gca()
    img_plt = plt.imshow(image)
    ax.set_facecolor([1.0, 1.0, 1.0])
    ax.axis("off")
    return img_plt


def draw_background(alpha_mask, background_color=[1.0, 1.0, 1.0]):
    """
    Draw a background image with transparency given by the alpha mask.
    """
    h, w = alpha_mask.shape[:2]
    C = np.zeros((h, w, 4))
    for ci in range(3):
        C[:, :, ci] = background_color[ci]
    C[:, :, 3] = 1.0 - alpha_mask
    plt.imshow(C)


def save_annotation_plot(
        annotation_set,
        with_exemplar=True,
        is_white=True,
        with_vf=False,
        out_file=None,
        white_factor=0.5
):
    """
    Save a visual plot of the annotation set with optional exemplar, background, and vector field.
    """
    fig_size = 10
    width_scale = fig_size / 10.0
    fig = plt.figure(figsize=(fig_size, fig_size))
    transparent = False
    ax = plt.subplot(1, 1, 1)

    if with_exemplar:
        annotation_set.plot_exemplar_image()
    else:
        empty_image = np.zeros_like(annotation_set.exemplar_image)
        plt.imshow(empty_image)
        ax.axis("off")
        transparent = True

    if with_vf:
        annotation_set.plot_orientations()

    if is_white:
        draw_background(white_factor * np.ones_like(annotation_set.alpha_mask))

    annotation_set.plot_annotations(width_scale=width_scale)

    save_fig(out_file, transparent=transparent)
    plt.close(fig)
