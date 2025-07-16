# -----------------------------------------------------------------------------
# Stroke Transfer for Participating Media
# http://cg.it.aoyama.ac.jp/yonghao/sig25/abstsig25.html
#
# File: util/model/out_weight.py
# Maintainer: Hideki Todo
#
# Description:
# Visualizing local weight maps from regression models.
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
import tqdm
from matplotlib import cm, pyplot as plt

from util.common.feature_basis_io import load_image, save_image
from util.model.orientation_model import VectorFieldRegressionModel
from util.plot.common import get_feature_mask
from util.regression_transfer.orientation import set_features, set_basis


def plot_local_weight_map_frame(
        vf_model: VectorFieldRegressionModel,
        feature_file_templates, basis_smooth_file_templates,
        resolution,
        transmittance_file_template,
        out_fig_file_template,
        frame,
        relocator=None
):
    """
    Plot and save a local weight map for a single frame.

    Args:
        vf_model: Vector field regression model instance with compute_weight_map() method.
        feature_file_templates (dict): Templates for feature file paths.
        basis_smooth_file_templates (dict): Templates for basis file paths.
        resolution (tuple): Target resolution for resizing (width, height).
        transmittance_file_template (str): Template for transmittance mask file.
        out_fig_file_template (str): Output path template for the figure.
        frame (int): Frame index to process.
        relocator (optional): Optional relocator object to remap features.
    """
    set_features(vf_model, feature_file_templates, frame, resolution, relocator=relocator)
    set_basis(vf_model, basis_smooth_file_templates, frame, resolution)

    mask = get_feature_mask(transmittance_file_template % frame, resolution=resolution)

    vf_model.set_mask(mask)

    weights = vf_model.compute_local_weight_map()

    for key in weights.keys():
        W = weights[key]

        fig = plt.figure(figsize=(16, 16))
        fig.tight_layout()
        ax = plt.subplot(1, 1, 1)
        plt.imshow(W, cmap=cm.PuOr, vmin=-1.0, vmax=1.0, origin="lower")
        plt.axis("off")

        out_file = out_fig_file_template % frame

        os.makedirs(os.path.dirname(out_file), exist_ok=True)
        fig.savefig(out_file, bbox_inches="tight", pad_inches=0, transparent=True)

        plt.clf()
        plt.close()

        I = load_image(out_file)
        I = cv2.resize(I, resolution)

        I[:, :, 3] = mask

        save_image(out_file, I)


def plot_local_weight_map(
        vf_model: VectorFieldRegressionModel,
        feature_file_templates, basis_smooth_file_templates,
        resolution,
        transmittance_file_template,
        out_fig_file_template,
        frames,
        relocator=None
):
    """
    Plot and save local weight maps for multiple frames.

    Args:
        vf_model: Vector field regression model instance with compute_weight_map() method.
        feature_file_templates (dict): Templates for feature file paths.
        basis_smooth_file_templates (dict): Templates for basis file paths.
        resolution (tuple): Target resolution for resizing (width, height).
        transmittance_file_template (str): Template for transmittance mask file.
        out_fig_file_template (str): Output path template for the figure.
        frames (list of int): List of frame indices to process.
        relocator (optional): Optional relocator object to remap features.
    """
    for frame in tqdm.tqdm(frames, "Plot Local Weight Map"):
        plot_local_weight_map_frame(
            vf_model,
            feature_file_templates, basis_smooth_file_templates,
            resolution,
            transmittance_file_template,
            out_fig_file_template,
            frame,
            relocator=relocator
        )
