# -----------------------------------------------------------------------------
# Stroke Transfer for Participating Media
# http://cg.it.aoyama.ac.jp/yonghao/sig25/abstsig25.html
#
# File: p8_smoothing.py
# Maintainer: Hideki Todo
#
# Description:
# Apply temporal and spatial smoothing to transferred orientation fields.
# Used to reduce noise in stroke directions before stroke generation.
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
from util.basis.smoothing_orientations import smoothing_orientations
from util.infra.logger import log_phase, log_subsection
from util.pipeline.pipeline_decorator import deco_pipeline
from util.pipeline.time_logger import resume_log, log_timing


@deco_pipeline
def p8_smoothing(
        orientation_file_template="stroke/orientation/orientation_%03d.hdf5",
        smooth_orientation_file_template="stroke/smooth_orientation/smooth_orientation_%03d.hdf5",
        lambda_spatial=1.0 / 100000.0,
        lambda_temporal=1.0 / 100.0,
        frame_start=1,
        frame_end=173,
        frame_skip=3,
        run_smoothing=True,
        transmittance_file_template=None,
        intensity_l_file_template=None,
        frames=None,
        resolution=(512, 512),
        out_video_file=None,
        use_random_factor=True,
        random_factor=0.001,
        plot_file_templates=None,
        plot=False
):
    """ Apply temporal and spatial smoothing to orientation fields.

    Args:
        orientation_file_template (str): Template path for input orientation files.
        smooth_orientation_file_template (str): Template path for output smoothed orientation files.
        lambda_spatial (float): Spatial regularization weight.
        lambda_temporal (float): Temporal regularization weight.
        frame_start (int): Start frame index.
        frame_end (int): End frame index.
        frame_skip (int): Interval between processed frames.
        transmittance_file_template (str, optional): Path template for transmittance maps.
        intensity_l_file_template (str, optional): Path template for luminance maps.
        frames (list[int], optional): Specific frames to process. If None, uses regular range.
        resolution (tuple): Output resolution (width, height).
        out_video_file (str, optional): Path to output video of smoothed orientations.
        plot_file_templates (dict, optional): Dictionary of templates for output plot images.
        plot (bool): Whether to generate visualizations for the smoothed fields.
    """
    if frames is None:
        frames = range(frame_start, frame_end + 1, frame_skip)

    if run_smoothing:
        log_phase(f"Orientation Smoothing")
        resume_log()
        smoothing_orientations(orientation_file_template,
                               smooth_orientation_file_template,
                               transmittance_file_template=transmittance_file_template,
                               lambda_spatial=lambda_spatial,
                               lambda_temporal=lambda_temporal,
                               frame_start=frame_start,
                               frame_end=frame_end,
                               frame_skip=frame_skip,
                               use_random_factor=use_random_factor,
                               random_factor=random_factor,
                               frames=frames)
        log_timing("smoothing", "", num_frames=len(frames))

    if plot:
        log_subsection("Plot Smoothed Results (Per Frame) ")
        plot_orientation(
            smooth_orientation_file_template,
            frames,
            transmittance_file_template=transmittance_file_template,
            intensity_l_file_template=intensity_l_file_template,
            out_fig_file_template=plot_file_templates["smooth_orientation"],
            resolution=resolution,
            out_video_file=out_video_file,
        )
