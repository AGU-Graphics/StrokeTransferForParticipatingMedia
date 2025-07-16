# -----------------------------------------------------------------------------
# Stroke Transfer for Participating Media
# http://cg.it.aoyama.ac.jp/yonghao/sig25/abstsig25.html
#
# File: p3_preprocess_feature_basis.py
# Maintainer: Hideki Todo
#
# Description:
# Preprocessing pipeline for standardizing features and smoothing basis fields.
# Used in early-stage computations before stroke rendering and synthesis.
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

from util.basis.plot import plot_basis_summary
from util.basis.smoothing_basis import smoothing_basis
from util.features.plot import plot_feature_summary
from util.features.standardize import standardize_features, standardize_features_baseline
from util.infra.logger import log_phase
from util.pipeline.pipeline_decorator import deco_pipeline
from util.pipeline.time_logger import resume_log, log_timing
from util.reference_rendering.tone_map_rendering import generate_reference_rendering_from_rgb_features


def p3_standardize_features(
        raw_feature_file_templates,
        feature_file_templates,
        T_infinity=1.5,
        T_1=0.6,
        q=1.0,
        frame_start=1,
        frame_end=172,
        frame_skip=3,
        learn_frames=[169],
        run_standardization=True,
        frames=None,
        resolution=None,
        t_obj=5.0,
        plot_feature_file_template=None,
        use_baseline_features=False,
        plot=False,
):
    """Standardize features for a sequence of frames.

    Args:
        raw_feature_file_templates (dict): Input templates for raw feature files.
        feature_file_templates (dict): Output templates for standardized features.
        T_infinity (float): Max luminance used in tone-mapping.
        T_1 (float): Reference luminance.
        q (float): IQR scaling coefficient.
        frame_start (int): First frame index.
        frame_end (int): Last frame index.
        frame_skip (int): Frame step interval.
        learn_frames (list[int]): Frames used for regression.
        run_standardization (bool): Whether to run standardization.
        frames (list[int] or None): Explicit list of frames to process.
        resolution (tuple[int, int] or None): Resize features to this resolution.
        t_obj (float): Temporal scale.
        plot_feature_file_template (str or None): Template for saving plots.
        plot (bool): Whether to generate and save plots.
    """

    L_M = T_infinity
    theta_T = math.log((1 + T_1 / T_infinity) / (1 - T_1 / T_infinity))

    if frames is None:
        frames = list(range(frame_start, frame_end + 1, frame_skip))
    else:
        frames = list(frames)

    for frame in learn_frames:
        if not frame in frames:
            frames.append(frame)

    resume_log()

    if run_standardization:
        log_phase("Standardize Feature Values")
        if use_baseline_features:
            standardize_features_baseline(
                raw_feature_file_templates,
                feature_file_templates, q=q,
                frames=frames,
                t_obj=t_obj,
                resolution=resolution
            )
        else:
            standardize_features(
                raw_feature_file_templates,
                feature_file_templates,
                L_M=L_M, theta_T=theta_T, q=q,
                frames=frames,
                t_obj=t_obj
            )

    log_timing("prepare", "standardization", num_frames=len(frames))

    if plot:
        log_phase("Generate Feature Plot Images")
        plot_feature_summary(
            feature_file_templates,
            frames,
            out_fig_file_template=plot_feature_file_template,
            raw_data=False,
            with_title=True,
            with_range=False,
            vmin_features=None,
            vmax_features=None,
            resolution=resolution
        )


def p3_smoothing_basis(
        basis_file_templates,
        basis_smooth_file_templates,
        frame_start=1,
        frame_end=172,
        frame_skip=3,
        frames=None,
        plot=False,
        transmittance_file_template=None,
        lambda_spatial=1.0 / 10000.0,
        lambda_temporal=1.0 / 100.0,
        run_basis_smoothing=True,
        plot_basis_file_template=None,
        intensity_l_file_template=None,
        resolution=None
):
    """Smooth the orientation basis fields over space and time.

    Args:
        basis_file_templates (dict): Input templates for raw basis files.
        basis_smooth_file_templates (dict): Output templates for smoothed bases.
        frame_start (int): First frame index.
        frame_end (int): Last frame index.
        frame_skip (int): Frame step interval.
        frames (list[int] or None): Explicit frame list to process.
        plot (bool): Whether to generate and save visualizations.
        lambda_spatial (float): Spatial smoothing weight.
        lambda_temporal (float): Temporal smoothing weight.
        plot_basis_file_template (str or None): Template for saving plot images.
        transmittance_file_template (str or None): Template for transmittance maps.
        intensity_l_file_template (str or None): Template for luminance maps.
        resolution (tuple[int, int] or None): Target resolution.
    """
    if run_basis_smoothing:
        log_phase("Smoothing Basis")
        smoothing_basis(
            basis_file_templates,
            basis_smooth_file_templates,
            transmittance_file_template,
            lambda_spatial=lambda_spatial,
            lambda_temporal=lambda_temporal,
            frame_start=frame_start,
            frame_end=frame_end,
            frame_skip=frame_skip,
            run_smoothing=run_basis_smoothing,
            frames=frames
        )

    if plot:
        log_phase("Generate Basis Plot Images")
        plot_basis_summary(
            basis_smooth_file_templates,
            frames,
            out_fig_file_template=plot_basis_file_template,
            transmittance_file_template=transmittance_file_template,
            intensity_l_file_template=intensity_l_file_template,
            resolution=resolution
        )


def plt_reference_rendering(
        raw_feature_file_templates,
        plot_file_template=None,
        out_video_file=None,
        T_infinity=1.5,
        T_1=0.6,
        frame_start=1,
        frame_end=173,
        frame_skip=3,
        frame_rate=24.0,
        raw_data=False,
        bg_color=[0.0, 1.0, 0.0],
        with_bg=False,
        frames=None,
        transmittance_file_template=None,
        resolution=None
):
    """Generate reference rendering images from RGB feature values.


    Args:
        raw_feature_file_templates (dict): Input templates for raw RGB feature maps.
        plot_file_template (str or None): Output image filename template.
        out_video_file (str or None): Output video filename.
        T_infinity (float): Max luminance value for mapping.
        T_1 (float): Reference luminance.
        frame_start (int): First frame index.
        frame_end (int): Last frame index.
        frame_skip (int): Frame interval.
        frame_rate (float): Frame rate for output video.
        frames (list[int] or None): Explicit frame list.
        transmittance_file_template (str or None): Template for transmittance maps.
        resolution (tuple[int, int] or None): Image resolution.
    """

    log_phase("Generate Tone Map Rendering Images")
    L_M = T_infinity
    theta_T = math.log((1 + T_1 / T_infinity) / (1 - T_1 / T_infinity))

    if frames is None:
        frames = range(frame_start, frame_end + 1, frame_skip)

    generate_reference_rendering_from_rgb_features(
        raw_feature_file_templates=raw_feature_file_templates,
        L_M=L_M, theta_T=theta_T,
        frames=frames,
        out_image_file_template=plot_file_template,
        out_video_file=out_video_file,
        frame_rate=frame_rate,
        raw_data=raw_data,
        bg_color=bg_color,
        with_bg=with_bg,
        transmittance_file_template=transmittance_file_template,
        resolution=resolution
    )


@deco_pipeline
def p3_preprocess_feature_basis(
        raw_feature_file_templates,
        feature_file_templates,
        basis_file_templates,
        basis_smooth_file_templates,
        T_infinity=1.5,
        T_1=0.6,
        q=1.0,
        frame_start=1,
        frame_end=172,
        frame_skip=3,
        learn_frames=[169],
        run_standardization=True,
        frames=None,
        resolution=None,
        t_obj=5.0,
        plot_feature_file_template=None,
        use_baseline_features=False,
        plot=False,
        transmittance_file_template=None,
        lambda_spatial=1.0 / 10000.0,
        lambda_temporal=1.0 / 100.0,
        run_basis_smoothing=True,
        plot_basis_file_template=None,
        intensity_l_file_template=None,
        plot_tone_mapping_rendering_file_template=None,
        video_tone_mapping_rendering_file=None,
        frame_rate=24.0
):
    """Run the full preprocessing pipeline for feature and basis preparation.

    Args:
        raw_feature_file_templates (dict): Input templates for raw features.
        feature_file_templates (dict): Output templates for standardized features.
        basis_file_templates (dict): Input templates for basis fields.
        basis_smooth_file_templates (dict): Output templates for smoothed bases.
        T_infinity (float): Max luminance.
        T_1 (float): Reference luminance.
        q (float): IQR scaling coefficient.
        frame_start (int): First frame.
        frame_end (int): Last frame.
        frame_skip (int): Frame interval.
        learn_frames (list[int]): Frames used for learning normalization.
        frames (list[int] or None): List of target frames.
        resolution (tuple[int, int] or None): Resize target.
        t_obj (float): Temporal scale.
        plot_feature_file_template (str or None): Output path for feature plots.
        plot (bool): Whether to generate plots.
        transmittance_file_template (str or None): Template for transmittance maps.
        lambda_spatial (float): Spatial smoothness weight.
        lambda_temporal (float): Temporal smoothness weight.
        plot_basis_file_template (str or None): Output path for basis plots.
        intensity_l_file_template (str or None): Template for luminance maps.
        plot_tone_mapping_rendering_file_template (str or None): Template for rendered image output.
        video_tone_mapping_rendering_file (str or None): Output video filename.
        frame_rate (float): Frame rate for rendering.
    """
    p3_standardize_features(
        raw_feature_file_templates,
        feature_file_templates,
        T_infinity=T_infinity,
        T_1=T_1,
        q=q,
        frame_start=frame_start,
        frame_end=frame_end,
        frame_skip=frame_skip,
        learn_frames=learn_frames,
        run_standardization=run_standardization,
        frames=frames,
        resolution=resolution,
        t_obj=t_obj,
        plot_feature_file_template=plot_feature_file_template,
        use_baseline_features=use_baseline_features,
        plot=plot,
    )

    p3_smoothing_basis(
        basis_file_templates,
        basis_smooth_file_templates,
        frame_start=frame_start,
        frame_end=frame_end,
        frame_skip=frame_skip,
        frames=frames,
        plot=plot,
        transmittance_file_template=transmittance_file_template,
        lambda_spatial=lambda_spatial,
        lambda_temporal=lambda_temporal,
        run_basis_smoothing=run_basis_smoothing,
        plot_basis_file_template=plot_basis_file_template,
        intensity_l_file_template=intensity_l_file_template,
        resolution=resolution
    )

    if plot:
        plt_reference_rendering(
            raw_feature_file_templates,
            plot_file_template=plot_tone_mapping_rendering_file_template,
            out_video_file=video_tone_mapping_rendering_file,
            T_infinity=T_infinity,
            T_1=T_1,
            frame_start=frame_start,
            frame_end=frame_end,
            frame_skip=frame_skip,
            frame_rate=frame_rate,
            raw_data=False,
            bg_color=[0.0, 1.0, 0.0],
            with_bg=False,
            frames=frames,
            transmittance_file_template=transmittance_file_template,
            resolution=resolution
        )
