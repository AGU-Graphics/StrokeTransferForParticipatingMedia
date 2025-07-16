# -----------------------------------------------------------------------------
# Stroke Transfer for Participating Media
# http://cg.it.aoyama.ac.jp/yonghao/sig25/abstsig25.html
#
# File: p11_final_composite.py
# Maintainer: Hideki Todo
#
# Description:
# Composite rendered strokes with undercoat layer to fill gaps in stroke coverage.
# Uses color transfer result as background to produce final output frames and video.
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

from util.image.image2video import images_to_video
from util.multi_layer.composite_layers import composite_layers_frames
from util.pipeline.time_logger import resume_log, log_timing


def p11_final_composite_default(
        final_file_template,
        video_final_file,
        undercoat_filename_template,
        frame_start=140,
        frame_end=151,
        frame_skip=1,
        bg_color=np.array([0.0, 0.0, 0.0, 0.0]),
        resolution=(1024, 1024),
        frame_rate=24.0
):
    """Composite final rendered strokes with undercoat and export as video.

    Args:
        final_file_template (str): Template for input stroke-rendered images.
        video_final_file (str): Path to output video file of the stroke-rendered frames.
        undercoat_filename_template (str): Template for undercoat images (color transfer results).
        frame_start (int): Start frame index.
        frame_end (int): End frame index.
        frame_skip (int): Frame sampling interval.
        bg_color (np.ndarray): Background color in RGBA format (default: transparent black).
        resolution (tuple): Output resolution as (width, height).
        frame_rate (float): Frame rate for the output video.
    """

    out_final_comp_template = final_file_template.replace("final", "final_comp")
    out_video_final_comp_file = video_final_file.replace("final", "final_comp")

    p11_final_composite(
        final_file_template,
        out_final_comp_template,
        undercoat_filename_template,
        frame_start=frame_start,
        frame_end=frame_end,
        frame_skip=frame_skip,
        bg_color=bg_color,
        resolution=resolution,
        out_video_final_comp_file=out_video_final_comp_file,
        frame_rate=frame_rate
    )


def p11_final_composite(
        final_file_template,
        out_final_comp_template,
        undercoat_filename_template,
        frame_start=140,
        frame_end=151,
        frame_skip=1,
        bg_color=np.array([0.0, 0.0, 0.0, 0.0]),
        resolution=(1024, 1024),
        out_video_final_comp_file=None,
        frame_rate=24.0
):
    """Composite stroke-rendered images and undercoat layer into final output.

    This function blends the stroke rendering with the undercoat layer
    to fill in any stroke gaps using the color transfer background.

    Args:
        final_file_template (str): Template for input stroke-rendered images.
        out_final_comp_template (str): Template for output composited images.
        undercoat_filename_template (str): Template for undercoat images (used as background).
        frame_start (int): Start frame index.
        frame_end (int): End frame index.
        frame_skip (int): Interval for frame sampling.
        bg_color (np.ndarray): Background color in RGBA format (e.g., [0.0, 0.0, 0.0, 0.0]).
        resolution (tuple): Output resolution in pixels as (width, height).
        out_video_final_comp_file (str, optional): Output video path for composited frames.
        frame_rate (float): Frame rate for the output video.
    """

    resume_log()
    frames = range(frame_start, frame_end + 1, frame_skip)

    layer_file_templates = [
        undercoat_filename_template,
        final_file_template
    ]

    composite_layers_frames(layer_file_templates, out_final_comp_template,
                            bg_color=bg_color,
                            resolution=resolution,
                            frames=frames)

    log_timing("final_composite", num_frames=len(frames))

    if out_video_final_comp_file is not None:
        frames = range(frame_start, frame_end + 1, frame_skip)
        images_to_video(out_final_comp_template, out_video_final_comp_file, frame_rate=frame_rate, frames=frames)
