# -----------------------------------------------------------------------------
# Stroke Transfer for Participating Media
# http://cg.it.aoyama.ac.jp/yonghao/sig25/abstsig25.html
#
# File: util/basis/plot.py
# Maintainer: Hideki Todo
#
# Description:
# Visualization utilities for orientation and basis vector fields in 2D feature maps.
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
import math
import os

import cv2
import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm

from util.common.feature_basis_def import BASIS_NAMES, FEATURE_BASIS_SYMBOLS
from util.common.feature_basis_io import load_hdf5
from util.image.image2video import images_to_video
from util.infra.logger import getLogger, log_warning
from util.plot.common import plot_vector_field, get_feature_mask, get_bg_image
from util.regression_transfer.orientation import normalize_vector_image

logger = getLogger()

def setup_matplotlib():
    """Configures matplotlib global settings for font rendering."""
    plt.rcParams['font.family'] = 'Times New Roman'
    plt.rcParams['mathtext.fontset'] = 'cm'


def prepare_mask_and_bg(transmittance_file, intensity_l_file, resolution):
    """Prepares mask and background image for visualization.

        Args:
            transmittance_file (str): Path to the transmittance file.
            intensity_l_file (str or None): Path to the L-channel intensity file.
            resolution (Tuple[int, int]): Output resolution (width, height).

        Returns:
            Tuple[np.ndarray or None, np.ndarray or None]: The mask and background image.
        """
    mask = get_feature_mask(transmittance_file, resolution=resolution)
    bg_image = None
    if intensity_l_file is not None:
        bg_image = get_bg_image(transmittance_file, intensity_l_file, resolution=resolution)
    return mask, bg_image


def preprocess_vector_field(V, resolution, mask=None):
    """Resizes and normalizes a vector field, optionally applying a mask.

        Args:
            V (np.ndarray): Input vector field.
            resolution (Tuple[int, int]): Target resolution (width, height).
            mask (np.ndarray, optional): Optional binary mask.

        Returns:
            np.ndarray: Processed vector field.
        """
    V = cv2.resize(V, resolution)
    V = normalize_vector_image(V)
    if mask is not None:
        V = np.einsum("ijk,ij->ijk", V, mask)
    return V


def save_orientation_figure(V, bg_image, out_file, figsize=(6, 6)):
    """Saves a figure visualizing orientation vectors over an optional background image.

        Args:
            V (np.ndarray): Vector field to visualize.
            bg_image (np.ndarray or None): Optional background image.
            out_file (str): Output file path.
            figsize (Tuple[int, int]): Size of the matplotlib figure.
        """
    fig = plt.figure(figsize=figsize)
    if bg_image is not None:
        plt.imshow(bg_image, origin="lower")
    plot_vector_field(V)
    plt.axis("off")
    os.makedirs(os.path.dirname(out_file), exist_ok=True)
    fig.savefig(out_file, bbox_inches="tight", pad_inches=0.0)
    plt.close()


def plot_orientation(
        orientation_file_template,
        frames,
        transmittance_file_template=None,
        intensity_l_file_template=None,
        out_fig_file_template=None,
        resolution=(512, 512),
        out_video_file=None,
        frame_rate=24.0
):
    """Plots orientation fields for multiple frames and optionally creates a video.

        Args:
            orientation_file_template (str): Template path to orientation HDF5 files.
            frames (List[int]): List of frame indices to process.
            transmittance_file_template (str or None): Template path for transmittance images.
            intensity_l_file_template (str or None): Template path for intensity (L-channel) images.
            out_fig_file_template (str or None): Template path for output figure images.
            resolution (Tuple[int, int]): Output resolution (width, height).
            out_video_file (str or None): Path to output video file.
            frame_rate (float): Frame rate for video.
        """
    setup_matplotlib()
    if out_fig_file_template is None:
        out_fig_file_template = orientation_file_template.replace(".h5", ".png")

    for frame in tqdm(frames, desc="Plot"):
        trans_file = transmittance_file_template % frame if transmittance_file_template else None
        intensity_file = intensity_l_file_template % frame if intensity_l_file_template else None
        mask, bg_image = prepare_mask_and_bg(trans_file, intensity_file, resolution)

        orientation_file = orientation_file_template % frame

        V = load_hdf5(orientation_file)
        if V is None:
            continue

        V = preprocess_vector_field(V, resolution, mask)

        out_file = out_fig_file_template % frame
        save_orientation_figure(V, bg_image, out_file)

    if out_video_file is not None:
        images_to_video(out_fig_file_template, out_video_file, frame_rate=frame_rate, frames=frames)


def plot_each_basis_for_all(
        basis_file_templates,
        frames,
        transmittance_file_template=None,
        intensity_l_file_template=None,
        out_fig_file_templates=None,
        resolution=(512, 512)
):
    """Plots vector fields for all basis types across frames.

        Args:
            basis_file_templates (Dict[str, str]): Dictionary mapping basis names to file templates.
            frames (List[int]): List of frame indices to process.
            transmittance_file_template (str or None): Transmittance file template.
            intensity_l_file_template (str or None): Intensity (L-channel) file template.
            out_fig_file_templates (Dict[str, str] or None): Output file templates per basis.
            resolution (Tuple[int, int]): Output resolution.
        """
    if out_fig_file_templates is None:
        out_fig_file_templates = {}

        for basis_name in basis_file_templates:
            basis_file_template = basis_file_templates[basis_name]

            if basis_file_template is not None:
                fig_file_template = basis_file_template.replace(".h5", ".png")
                out_fig_file_templates[basis_name] = fig_file_template
            else:
                out_fig_file_templates[basis_name] = None

    for basis_name, basis_file_template in basis_file_templates.items():
        if basis_file_template is None:
            print(f"Skip {basis_name}")
            continue

        out_fig_file_template = out_fig_file_templates[basis_name]

        plot_orientation(
            orientation_file_template=basis_file_template,
            frames=frames,
            transmittance_file_template=transmittance_file_template,
            intensity_l_file_template=intensity_l_file_template,
            out_fig_file_template=out_fig_file_template,
            resolution=resolution,
            out_video_file=None,
            frame_rate=24.0
        )


def plot_basis_summary_frame(
        basis_file_templates,
        frame,
        out_fig_file_template,
        with_title=True,
        dense_plot=True,
        transmittance_file_template=None,
        intensity_l_file_template=None,
        resolution=(500, 500)):
    """Plots all basis vector fields in a grid layout for a single frame.

        Args:
            basis_file_templates (Dict[str, str]): Dictionary mapping basis names to file templates.
            frame (int): Frame index to visualize.
            out_fig_file_template (str): Output image file template.
            with_title (bool): Whether to display feature names as titles.
            dense_plot (bool): Whether to use a dense grid layout.
            transmittance_file_template (str): Template path to transmittance image.
            intensity_l_file_template (str): Template path to intensity L-channel image.
            resolution (Tuple[int, int]): Output resolution (width, height).
        """

    setup_matplotlib()

    mask = get_feature_mask(transmittance_file_template % frame, resolution=resolution)
    bg_image = get_bg_image(transmittance_file_template % frame,
                            intensity_l_file_template % frame,
                            resolution=resolution)

    if mask is None:
        log_warning(logger, f"No mask found for frame {frame}, skipping.")
        return

    if dense_plot:
        cols = 4
    else:
        cols = 2
    rows = int(math.ceil(len(BASIS_NAMES) / cols))

    aspect = bg_image.shape[0] / bg_image.shape[1]

    aspect += 0.1

    tex_size = 25

    fig = plt.figure(figsize=(cols * 4, rows * aspect * 4))
    plt.subplots_adjust(left=0.0, bottom=0.0, right=1.0, top=1.0, wspace=0.02, hspace=None)

    for i, basis_name in enumerate(BASIS_NAMES):
        basis_file_template = basis_file_templates[basis_name]
        if basis_file_template is None:
            continue

        basis_file = basis_file_template % frame
        if not os.path.exists(basis_file):
            raise FileNotFoundError(basis_file)

        V = load_hdf5(basis_file)
        V = V[:, :, :2]
        V = cv2.resize(V, resolution)

        V = normalize_vector_image(V)
        V = np.einsum("ijk,ij->ijk", V, mask)
        plt.subplot(rows, cols, i + 1)
        # plot_orientation_color(V, add_white=0.0)
        plt.imshow(bg_image, origin="lower")
        plot_vector_field(V)
        if with_title:
            title = FEATURE_BASIS_SYMBOLS[basis_name]
            ax = plt.gca()
            plt.text(0.5, -0.1, title, ha='center', va='center', fontsize=tex_size, transform=ax.transAxes)

        plt.xticks([])
        plt.yticks([])

    out_file = out_fig_file_template % frame

    out_dir = os.path.dirname(out_file)
    os.makedirs(out_dir, exist_ok=True)
    fig.savefig(out_file, bbox_inches="tight", pad_inches=0.0, transparent=False)
    plt.close()


def plot_basis_summary(
        basis_file_templates,
        frames,
        out_fig_file_template=None,
        with_title=True,
        dense_plot=True,
        transmittance_file_template=None,
        intensity_l_file_template=None,
        resolution=(500, 500)):
    for frame in tqdm(frames, desc="Plot Basis"):
        plot_basis_summary_frame(
            basis_file_templates=basis_file_templates,
            frame=frame,
            out_fig_file_template=out_fig_file_template,
            with_title=with_title,
            dense_plot=dense_plot,
            transmittance_file_template=transmittance_file_template,
            intensity_l_file_template=intensity_l_file_template,
            resolution=resolution
        )
