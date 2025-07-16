# -----------------------------------------------------------------------------
# Stroke Transfer for Participating Media
# http://cg.it.aoyama.ac.jp/yonghao/sig25/abstsig25.html
#
# File: util/features/plot.py
# Maintainer: Hideki Todo
#
# Description:
# This module provides visualization tools for feature maps extracted from HDF5 files.
# It supports per-feature plotting as well as grid-based summaries across multiple frames.
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
import numpy as np
from matplotlib import pyplot as plt, cm as cm
from tqdm import tqdm

from util.common.feature_basis_def import RAW_FEATURE_NAMES, FEATURE_NAMES, FEATURE_BASIS_SYMBOLS
from util.common.feature_basis_io import load_hdf5, load_rgba, save_rgba

# === Global Config ===
COLORMAP = cm.plasma


def setup_matplotlib():
    """Configures matplotlib global settings for font rendering."""
    plt.rcParams['font.family'] = 'Times New Roman'
    plt.rcParams['mathtext.fontset'] = 'cm'


def compute_value_range(feature_file_template,
                        frames, mask=None):
    """Computes the min and max values across feature maps for specified frames.

        Args:
            feature_file_template (str): Template path to the HDF5 feature files (e.g., '..._%03d.h5').
            frames (List[int]): List of frame indices to evaluate.
            mask (np.ndarray, optional): Mask image to restrict value computation to certain areas.

        Returns:
            Tuple[float, float]: (vmin, vmax) range of feature values.
        """
    vmax = np.finfo(np.float32).min
    vmin = np.finfo(np.float32).max
    for frame in frames:
        feature_file = feature_file_template % frame
        I = load_hdf5(feature_file)
        if mask is not None:
            I = I[mask > 0.5]
        vmax = max(vmax, np.max(I))
        vmin = min(vmin, np.min(I))

    return vmin, vmax


def resize_rgba_image(file_path, resolution):
    """Resizes an RGBA image to the given resolution and saves it back.

        Args:
            file_path (str): Path to the RGBA image file to be resized.
            resolution (Tuple[int, int]): Target resolution as (width, height).
        """
    image = load_rgba(file_path)
    if image is not None:
        image = cv2.resize(image, resolution)
        save_rgba(file_path, image)


def plot_each_feature_for_all(feature_file_templates,
                              frames,
                              out_fig_file_templates=None,
                              raw_data=False,
                              vmin=None,
                              vmax=None):
    """Plots each feature independently across all frames and saves the result as image files.

        Args:
            feature_file_templates (Dict[str, str]): Dictionary mapping feature names to file templates.
            frames (List[int]): List of frame indices to process.
            out_fig_file_templates (Dict[str, str], optional): Output path templates for each feature.
            raw_data (bool): Whether to visualize raw feature values with colorbar and vmin/vmax.
            vmin (float, optional): Minimum value for color scaling (used if raw_data is True).
            vmax (float, optional): Maximum value for color scaling (used if raw_data is True).
        """
    setup_matplotlib()

    if out_fig_file_templates is None:
        out_fig_file_templates = {}

        for feature_name in feature_file_templates:
            feature_file_template = feature_file_templates[feature_name]

            if feature_file_template is not None:
                fig_file_template = feature_file_template.replace(".h5", ".png")
                out_fig_file_templates[feature_name] = fig_file_template
            else:
                out_fig_file_templates[feature_name] = None

    for feature_name in feature_file_templates:
        if feature_name == "bounding_box":
            continue

        feature_file_template = feature_file_templates[feature_name]
        if feature_file_template is None:
            print(f"Skip {feature_name}")
            continue

        if raw_data:
            if vmin is None:
                vmin, vmax = compute_value_range(feature_file_template, frames)

        print(f"plot {feature_name}")

        for frame in frames:
            feature_file = feature_file_template % frame

            I = load_hdf5(feature_file)

            h, w = I.shape[:2]
            resolution = (w, h)

            fig = plt.figure(figsize=(6, 6))
            plt.subplot(1, 1, 1)
            if raw_data:
                plt.imshow(I, cmap=COLORMAP, origin="lower", vmin=vmin, vmax=vmax)
                plt.colorbar()
            else:
                plt.imshow(I, vmin=-1, vmax=1, cmap=COLORMAP, origin="lower")

            plt.axis("off")

            out_file = out_fig_file_templates[feature_name] % frame
            out_dir = os.path.dirname(out_file)
            os.makedirs(out_dir, exist_ok=True)
            fig.savefig(out_file, bbox_inches="tight", pad_inches=0.0, transparent=False)
            plt.close()

            if not raw_data:
                resize_rgba_image(out_file, resolution)


def plot_feature_summary(
        feature_file_templates,
        frames,
        out_fig_file_template=None,
        raw_data=False,
        with_title=True,
        with_range=False,
        vmin_features=None,
        vmax_features=None,
        resolution=None
):
    """Plots a summary image of all features in a grid for each frame.

        Args:
            feature_file_templates (Dict[str, str]): Dictionary mapping feature names to file templates.
            frames (List[int]): List of frame indices to process.
            out_fig_file_template (str, optional): Output image file template for each summary image.
            raw_data (bool): Whether to use raw feature values with automatic range scaling.
            with_title (bool): Whether to include symbolic feature names as titles.
            with_range (bool): Whether to append min/max value range in titles.
            vmin_features (Dict[str, float], optional): Precomputed minimum values for each feature.
            vmax_features (Dict[str, float], optional): Precomputed maximum values for each feature.
            resolution (Tuple[int, int], optional): Optional resolution to adjust figure aspect ratio.
        """
    setup_matplotlib()

    if raw_data:
        feature_names = [feature_name for feature_name in RAW_FEATURE_NAMES if feature_name != "bounding_box"]
    else:
        feature_names = [feature_name for feature_name in FEATURE_NAMES if feature_name != "bounding_box"]

    if raw_data:
        vmin_features = {}
        vmax_features = {}

        for feature_name, feature_file_template in feature_file_templates.items():
            if feature_name == "bounding_box":
                continue
            if feature_file_template is None:
                continue
            vmin, vmax = compute_value_range(feature_file_template, frames)
            vmin_features[feature_name] = vmin
            vmax_features[feature_name] = vmax

    cols = 5
    rows = int(math.ceil(len(feature_names) / cols))

    tex_size = 25

    if resolution is not None:
        aspect = resolution[1] / resolution[0]
    else:
        aspect = 1.0

    aspect += 0.1

    for frame in tqdm(frames, desc="Plot Features"):
        fig = plt.figure(figsize=(cols * 4, rows * aspect * 4))
        plt.subplots_adjust(left=0.0, bottom=0.0, right=1.0, top=1.0, wspace=0.02, hspace=None)

        for i, feature_name in enumerate(feature_names):
            feature_file_template = feature_file_templates[feature_name]
            if feature_file_template is None:
                continue

            feature_file = feature_file_template % frame

            if not os.path.exists(feature_file):
                raise FileNotFoundError(feature_file)

            I = load_hdf5(feature_file)

            plt.subplot(rows, cols, i + 1)
            if raw_data:
                plt.imshow(I, cmap=COLORMAP, origin="lower")
                plt.colorbar()
                plot_vmin = None
                plot_vmax = None
                if vmin_features is not None:
                    plot_vmin = vmin_features[feature_name]
                if vmax_features is not None:
                    plot_vmax = vmax_features[feature_name]
                plt.clim(plot_vmin, plot_vmax)

                if with_title:
                    title = FEATURE_BASIS_SYMBOLS[feature_name]
                    if with_range:
                        title += f": [{np.min(I):.2f}, {np.max(I):.2f}]"
                    ax = plt.gca()
                    plt.text(0.5, -0.1, title, ha='center', va='center', fontsize=tex_size, transform=ax.transAxes)
            else:
                plt.imshow(I, vmin=-1, vmax=1, cmap=COLORMAP, origin="lower")

                if with_title:
                    title = FEATURE_BASIS_SYMBOLS[feature_name]
                    if with_range:
                        title += f": [{np.min(I):.2f}, {np.max(I):.2f}]"

                    ax = plt.gca()
                    plt.text(0.5, -0.1, title, ha='center', va='center', fontsize=tex_size, transform=ax.transAxes)

            plt.axis("off")

        out_file = out_fig_file_template % frame

        out_dir = os.path.dirname(out_file)
        os.makedirs(out_dir, exist_ok=True)
        fig.savefig(out_file, bbox_inches="tight", pad_inches=0.0, transparent=False)

        plt.close()
