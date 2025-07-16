# -----------------------------------------------------------------------------
# Stroke Transfer for Participating Media
# http://cg.it.aoyama.ac.jp/yonghao/sig25/abstsig25.html
#
# File: p9_gen_strokes.py
# Maintainer: Hideki Todo
#
# Description:
# Generate final stroke data (HDF5) using transferred and smoothed attributes.
# Wraps cpp_tools/bin/gen_strokes_cli to synthesize strokes from orientation, color, width, and length.
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

from util.image.image2video import images_to_video
from util.pipeline.time_logger import gen_log_file_path
from util.pipeline.time_logger import resume_log, log_timing
from util.stroke_rendering.cpp_stroke_pipelines import gen_strokes


def p9_gen_strokes(
        attribute_file_templates,
        out_stroke_file_templates,
        frame_start=140,
        frame_end=141,
        frame_skip=1,
        angular_random_offset_deg=5.0,
        random_offset_factor=1.0,
        length_factor=1.0,
        width_factor=1.0,
        length_random_factor_relative=0.0,
        width_random_factor_relative=0.0,
        sort_type="add_sort_index",
        clip_with_undercoat_alpha=0,
        texture_filename="../../textures/texture.png",
        texture_for_active_set_for_new_stroke_filename="../../textures/texture_for_new.png",
        texture_for_active_set_for_existing_stroke_filename="../../textures/texture_for_existing.png",
        num_textures=1,
        texture_length_mipmap_level=1,
        frame_rate=24.0,
        resolution=(1024, 1024),
        mask_file_template="",
        active_set_file_template="",
        remove_hidden_strokes=1,
        remove_hidden_strokes_thr_contribution=1.0 / 256.0,
        remove_hidden_strokes_thr_alpha=1.0 / 256.0,
        stroke_dir="./",
        velocity_u_file_template="",
        velocity_v_file_template="",
        stroke_step_length=1.0 / 256.0,
        stroke_step_length_accuracy=0.1,
        consecutive_failure_max=100,
        out_video_stroke_file=None,
        log_file=None,
        region_label_file_template=None
):
    """ Generate stroke data using transferred and smoothed attributes.

    This function wraps the `gen_strokes_cli` C++ binary to synthesize stylized strokes
    based on input fields such as orientation, color, width, and length.

    Args:
        attribute_file_templates (dict): Dictionary of file templates for input attributes
            (keys: "smooth_orientation", "undercoat", "color", "width", "length").
        out_stroke_file_templates (dict): Dictionary of output file templates
            (keys: "anchor", "stroke", "stroke_data").
        frame_start (int): Start frame index.
        frame_end (int): End frame index.
        frame_skip (int): Frame interval.
        angular_random_offset_deg (float): Random angle perturbation in degrees.
        random_offset_factor (float): Scaling factor for positional randomness.
        length_factor (float): Global scale multiplier for stroke length.
        width_factor (float): Global scale multiplier for stroke width.
        length_random_factor_relative (float): Relative randomness in length.
        width_random_factor_relative (float): Relative randomness in width.
        sort_type (str): Method for sorting strokes (e.g., "add_sort_index").
        clip_with_undercoat_alpha (int): Whether to use undercoat alpha for clipping.
        texture_filename (str): Path to stroke texture image.
        texture_for_active_set_for_new_stroke_filename (str): Texture for new strokes.
        texture_for_active_set_for_existing_stroke_filename (str): Texture for existing strokes.
        num_textures (int): Number of textures used.
        texture_length_mipmap_level (int): Mipmap level for texture-length mapping.
        frame_rate (float): Frame rate for output video.
        resolution (tuple): Output image resolution.
        mask_file_template (str): Optional mask file template.
        active_set_file_template (str): Optional template for stroke active sets.
        remove_hidden_strokes (int): Whether to remove strokes occluded by undercoat.
        remove_hidden_strokes_thr_contribution (float): Visibility threshold for stroke contribution.
        remove_hidden_strokes_thr_alpha (float): Alpha threshold for visibility.
        stroke_dir (str): Output directory for strokes.
        velocity_u_file_template (str): Velocity u-component file template.
        velocity_v_file_template (str): Velocity v-component file template.
        stroke_step_length (float): Step size for stroke integration.
        stroke_step_length_accuracy (float): Accuracy threshold for step computation.
        consecutive_failure_max (int): Max failures allowed in stroke generation.
        out_video_stroke_file (str): Path to output video file for strokes.
        log_file (str, optional): Log file path. If None, auto-generated.
        region_label_file_template (str, optional): Region label template for selective processing.
    """

    if log_file is None:
        out_dir = os.path.dirname(os.path.dirname(out_stroke_file_templates["stroke"]))
        log_name = os.path.join(out_dir, "log", "p9_gen_strokes")
        log_file = gen_log_file_path(log_name)

    resume_log()
    gen_strokes(
        frame_start=frame_start,
        frame_end=frame_end,
        frame_skip=frame_skip,
        angular_random_offset_deg=angular_random_offset_deg,
        random_offset_factor=random_offset_factor,
        length_factor=length_factor,
        width_factor=width_factor,
        length_random_factor_relative=length_random_factor_relative,
        width_random_factor_relative=width_random_factor_relative,
        sort_type=sort_type,
        clip_with_undercoat_alpha=clip_with_undercoat_alpha,
        texture_filename=texture_filename,
        texture_for_active_set_for_new_stroke_filename=texture_for_active_set_for_new_stroke_filename,
        texture_for_active_set_for_existing_stroke_filename=texture_for_active_set_for_existing_stroke_filename,
        num_textures=num_textures,
        texture_length_mipmap_level=texture_length_mipmap_level,
        out_anchor_filename_template=out_stroke_file_templates["anchor"],
        out_stroke_filename_template=out_stroke_file_templates["stroke"],
        out_stroke_data_filename_template=out_stroke_file_templates["stroke_data"],
        orientation_filename_template=attribute_file_templates["smooth_orientation"],
        undercoat_filename_template=attribute_file_templates["undercoat"],
        color_filename_template=attribute_file_templates["color"],
        width_filename_template=attribute_file_templates["width"],
        length_filename_template=attribute_file_templates["length"],
        frame_rate=frame_rate,
        resolution=resolution,
        mask_file_template=mask_file_template,
        active_set_file_template=active_set_file_template,
        remove_hidden_strokes=remove_hidden_strokes,
        remove_hidden_strokes_thr_contribution=remove_hidden_strokes_thr_contribution,
        remove_hidden_strokes_thr_alpha=remove_hidden_strokes_thr_alpha,
        stroke_dir=stroke_dir,
        vx_filename_template=velocity_u_file_template,
        vy_filename_template=velocity_v_file_template,
        stroke_step_length=stroke_step_length,
        stroke_step_length_accuracy=stroke_step_length_accuracy,
        consecutive_failure_max=consecutive_failure_max,
        log_file=log_file,
        region_label_file_template=region_label_file_template
    )
    log_timing("gen_strokes", num_frames=len(range(frame_start, frame_end + 1, frame_skip)))

    if out_video_stroke_file is not None:
        frames = range(frame_start, frame_end + 1, frame_skip)
        images_to_video(out_stroke_file_templates["stroke"], out_video_stroke_file, frame_rate=frame_rate,
                        frames=frames)
