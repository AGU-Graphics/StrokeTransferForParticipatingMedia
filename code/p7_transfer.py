# -----------------------------------------------------------------------------
# Stroke Transfer for Participating Media
# http://cg.it.aoyama.ac.jp/yonghao/sig25/abstsig25.html
#
# File: p7_transfer.py
# Maintainer: Hideki Todo
#
# Description:
# Generate stroke attributes via transfer using regression models with feature and basis inputs.
# Produces orientation, color (with optional relocation), width, and length.
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
from util.basis.plot import plot_orientation
from util.model.model_io import load_model
from util.pipeline.pipeline_decorator import deco_pipeline
from util.pipeline.time_logger import resume_log, pause_log, log_timing
from util.regression_transfer.color_width_length import p7_color_width_length_transfer
from util.regression_transfer.orientation import p7_orientation_transfer
from util.relocation.feature_relocator import load_feature_dependent_relocator


@deco_pipeline
def p7_transfer(
        model_files,
        feature_file_templates,
        basis_smooth_file_templates,
        out_attribute_file_templates,
        out_video_attribute_files,
        frame_start=1,
        frame_end=172,
        frame_skip=3,
        resolution=(512, 512),
        frames=None,
        relocator_setting_file=None,
        transmittance_file_template=None,
        intensity_l_file_template=None,
        with_color_bar=False,
        plot_file_templates=None,
        plot=False,
        transfer_targets=None
):
    """ Transfer stroke attributes using pre-trained regression models.

    Args:
        model_files (dict): Dictionary mapping attribute names to model file paths.
        feature_file_templates (dict): Templates for input feature files.
        basis_smooth_file_templates (dict): Templates for smoothed basis field files.
        out_attribute_file_templates (dict): Templates for output attribute files.
        out_video_attribute_files (dict): Paths for output attribute video files.
        frame_start (int): Start frame index.
        frame_end (int): End frame index.
        frame_skip (int): Frame step size.
        resolution (tuple): Target resolution (width, height).
        frames (list[int], optional): Specific frames to process. If None, uses range.
        relocator_setting_file (str, optional): Path to relocator setting file for color transfer.
        transmittance_file_template (str, optional): Template for transmittance input.
        intensity_l_file_template (str, optional): Template for luminance input.
        with_color_bar (bool): Whether to add color bar to visualizations.
        plot_file_templates (dict, optional): Templates for plot output images.
        plot (bool): Whether to generate visualization plots.
        transfer_targets (list[str] or None): If specified, only run transfer for selected attributes
            (e.g., ["orientation", "color", "width", "length"]).

    """
    run_orientation = transfer_targets is None or "orientation" in transfer_targets
    run_color = transfer_targets is None or "color" in transfer_targets
    run_width = transfer_targets is None or "width" in transfer_targets
    run_length = transfer_targets is None or "length" in transfer_targets

    if frames is None:
        frames = list(range(frame_start, frame_end + 1, frame_skip))

    if relocator_setting_file is not None:
        relocator = load_feature_dependent_relocator(relocator_setting_file)
    else:
        relocator = None

    if run_orientation:
        model = load_model(model_files["orientation"])

        resume_log()
        p7_orientation_transfer(
            model,
            feature_file_templates,
            basis_smooth_file_templates,
            frames=frames,
            resolution=resolution,
            relocator=relocator,
            out_orientation_file_template=out_attribute_file_templates["orientation"])
        log_timing("transfer", "orientation transfer", num_frames=len(frames))
        pause_log()

        if plot:
            out_fig_file_template = plot_file_templates["orientation"]

            plot_orientation(
                out_attribute_file_templates["orientation"],
                frames,
                transmittance_file_template=transmittance_file_template,
                intensity_l_file_template=intensity_l_file_template,
                out_fig_file_template=out_fig_file_template,
                resolution=resolution,
                out_video_file=out_video_attribute_files["orientation"]
            )

    color_model = None
    width_model = None
    length_model = None

    if run_color:
        color_model = load_model(model_files["color"])

    if run_width:
        width_model = load_model(model_files["width"])

    if run_length:
        length_model = load_model(model_files["length"])

    p7_color_width_length_transfer(color_model, width_model, length_model,
                                   feature_file_templates,
                                   out_attribute_file_templates,
                                   out_video_attribute_files,
                                   frames, resolution=resolution,
                                   relocator=relocator,
                                   transmittance_file_template=transmittance_file_template,
                                   with_color_bar=with_color_bar,
                                   out_plot_file_attributes=plot_file_templates,
                                   plot=plot,
                                   transfer_targets=transfer_targets
                                   )
