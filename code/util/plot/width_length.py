# -----------------------------------------------------------------------------
# Stroke Transfer for Participating Media
# http://cg.it.aoyama.ac.jp/yonghao/sig25/abstsig25.html
#
# File: util/plot/width_length.py
# Maintainer: Hideki Todo
#
# Description:
# Visualizes scalar features with optional masking and video export.
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
import matplotlib.cm as cm
import matplotlib.pyplot as plt
import numpy as np
import tqdm

from util.common.feature_basis_io import load_rgba, load_image, save_rgba
from util.image.image2video import images_to_video
from util.plot.common import get_feature_mask

colormap = cm.plasma


def compute_value_range(
        input_file_template="stroke_relocate/width/width_%03d.hdf5",
        frames=[1]):
    """Compute min and max values of scalar features across frames.

    Args:
        input_file_template (str): Template path for input files.
        frames (list[int]): List of frame indices to read.

    Returns:
        tuple[float, float]: (vmin, vmax)
    """
    vmax = -np.inf
    vmin = np.inf
    for frame in frames:
        input_file = input_file_template % frame
        I = load_image(input_file)
        vmax = max(vmax, np.max(I))
        vmin = min(vmin, np.min(I))

    return vmin, vmax


def plot_scalar_feature_frame(
        input_file_template,
        frame,
        output_file_template=None,
        resolution=(512, 512),
        vmin=None, vmax=None,
        with_color_bar=False,
        transmittance_file_template=None,
        show_plot=False
):
    """Plot a single scalar feature frame with optional masking and color bar.

    Args:
        input_file_template (str): Template for input file path.
        frame (int): Frame index to plot.
        output_file_template (str, optional): Template for saving the output image.
        resolution (tuple[int, int]): Target output resolution.
        vmin (float, optional): Min value for colormap.
        vmax (float, optional): Max value for colormap.
        with_color_bar (bool): Whether to include a color bar.
        transmittance_file_template (str, optional): Template for transmittance mask.
        show_plot (bool): If True, display plot interactively.
    """
    input_file = input_file_template % frame
    I = load_image(input_file)

    if transmittance_file_template is not None:
        mask = get_feature_mask(transmittance_file_template % frame, resolution=resolution)
        I = np.einsum("ij,ij->ij", I, mask)

    aspect = resolution[1] / resolution[0]

    fig = plt.figure(figsize=(6, 6 * aspect))
    ax = plt.subplot(1, 1, 1)
    plt.imshow(I, origin="lower", cmap=colormap, vmin=vmin, vmax=vmax)
    if with_color_bar:
        plt.colorbar()
    plt.xticks([])
    plt.yticks([])

    if show_plot:
        plt.show()
    else:
        out_file = output_file_template % frame
        out_dir = os.path.dirname(out_file)

        os.makedirs(out_dir, exist_ok=True)
        fig.savefig(out_file, bbox_inches="tight", pad_inches=0.0, transparent=False)
        plt.close()

        if not with_color_bar:
            I = load_rgba(out_file)
            I = cv2.resize(I, resolution)
            save_rgba(out_file, I)


def plot_scalar_feature_frames(
        input_file_template,
        frames,
        output_file_template=None,
        resolution=(512, 512),
        vmin=None, vmax=None,
        with_color_bar=False,
        transmittance_file_template=None,
        out_video_file=None,
        frame_rate=24.0,
        show_plot=False
):
    """Plot a sequence of scalar feature frames and optionally create a video.

    Args:
        input_file_template (str): Template for input file paths.
        frames (list[int]): List of frame indices to process.
        output_file_template (str, optional): Template for saving output images.
        resolution (tuple[int, int]): Output resolution.
        vmin (float, optional): Min value for colormap.
        vmax (float, optional): Max value for colormap.
        with_color_bar (bool): Whether to include color bars.
        transmittance_file_template (str, optional): Template for transmittance mask files.
        out_video_file (str, optional): Path to save the output video.
        frame_rate (float): Frame rate for the video.
        show_plot (bool): If True, display plots instead of saving.
    """
    if vmin is None or vmax is None:
        vmin, vmax = compute_value_range(
            input_file_template=input_file_template,
            frames=frames)

    if output_file_template is None:
        output_file_template = input_file_template.replace(".h5", ".png")

    for frame in tqdm.tqdm(frames, desc="Plot"):
        plot_scalar_feature_frame(
            input_file_template,
            frame,
            output_file_template=output_file_template,
            resolution=resolution,
            vmin=vmin, vmax=vmax,
            with_color_bar=with_color_bar,
            transmittance_file_template=transmittance_file_template,
            show_plot=show_plot
        )

    if out_video_file is not None:
        images_to_video(output_file_template, out_video_file, frame_rate=frame_rate, frames=frames)
