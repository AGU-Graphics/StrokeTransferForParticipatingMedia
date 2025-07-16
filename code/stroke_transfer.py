# -----------------------------------------------------------------------------
# Stroke Transfer for Participating Media
# http://cg.it.aoyama.ac.jp/yonghao/sig25/abstsig25.html
#
# File: stroke_transfer.py
# Maintainer: Hideki Todo and Naoto Shirashima
#
# Description:
# Defines pipeline entry points for stroke transfer processing.
# Invoked from output/run.py to execute selected or grouped pipeline stages based on settings.
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
import json
import logging
import os

from aux_exemplar_frame_estimation import aux_exemplar_frame_estimation
from aux_transfer_region_labels import aux_transfer_region_labels
from p2_compute_features import p2_compute_features
from p3_preprocess_feature_basis import p3_preprocess_feature_basis, plt_reference_rendering
from p4_annotation_interpolation import p4_annotation_interpolation
from p5_regression import p5_regression
from p6_relocation import p6_relocation
from p7_transfer import p7_transfer
from p8_smoothing import p8_smoothing
from p9_gen_strokes import p9_gen_strokes
from p10_render_strokes import p10_render_strokes
from p10_render_strokes_pencil import p10_render_strokes_pencil
from p11_final_composite import p11_final_composite
from util.infra.logger import log_info, getLogger, set_level
from util.pipeline.time_logger import init_log, close_log

logger = getLogger()

PIPELINE_LIST = {
    "p2_compute_features": "Feature computation from volume data",
    "p3_preprocess_feature_basis": "Feature standardization, smoothing basis",
    "p4_annotation_interpolation": "Annotation interpolation",
    "p5_regression": "Regression",
    "p6_relocation": "Relocation",
    "p7_transfer": "Transfer",
    "p8_smoothing": "Orientation smoothing",
    "p9_gen_strokes": "Stroke generation",
    "p10_render_strokes": "Stroke rendering",
    "p10_render_strokes_pencil": "Pencil-style rendering (optional)",
    "p11_final_composite": "Final composite",
    "aux_exemplar_frame_estimation": "Exemplar frame estimation (aux)",
    "aux_transfer_region_labels": "Transfer region labels (aux)",
    "plt_reference_rendering": "Visualizing reference rendering (plt)"
}

PIPELINE_GROUPS = {
    "exemplar_frame_estimation": [
        "p2_compute_features",
        "p3_preprocess_feature_basis",
        "aux_exemplar_frame_estimation"
    ],
    "prepare_feature_basis": [
        "p2_compute_features",
        "p3_preprocess_feature_basis",
    ],
    "regression_transfer": [
        "p4_annotation_interpolation",
        "p5_regression",
        "p6_relocation",
        "p7_transfer",
        "p8_smoothing",
        "aux_transfer_region_labels"
    ],
    "stroke_rendering": [
        "p9_gen_strokes",
        "p10_render_strokes",
        "p10_render_strokes_pencil",
        "p11_final_composite"
    ],
}


def list_available_pipelines():
    """
    Print all available individual pipeline stages defined in the system.
    """
    print("Available pipelines:")
    for name in PIPELINE_LIST.keys():
        print(f"  {name:30} : {PIPELINE_LIST[name]}")


def list_pipeline_groups():
    """
    Print all available pipeline groups and their corresponding stages.
    """
    print("Available pipeline groups:")
    for name, stages in PIPELINE_GROUPS.items():
        print(f"  {name:30} : {', '.join(stages)}")


def expand_pipeline_groups(pipelines):
    """
    Expand pipeline group names into individual pipeline stage names.

    Args:
        pipelines (list[str] or None): List of pipeline stage names or group names.

    Returns:
        list[str] or None: Expanded list of pipeline stage names.
    """
    if pipelines is None:
        return None
    expanded = []
    for p in pipelines:
        if p in PIPELINE_GROUPS:
            expanded.extend(PIPELINE_GROUPS[p])
        else:
            expanded.append(p)
    return expanded


def load_pipeline_settings(pipeline_setting_file):
    """
    Load JSON-based pipeline settings from the specified file.

    Args:
        pipeline_setting_file (str): Path to the JSON settings file.

    Returns:
        dict or None: Parsed settings dictionary if successful, otherwise None.
    """
    try:
        with open(pipeline_setting_file, "r") as f:
            return json.load(f)
    except Exception as e:
        print(f"[ERROR] Failed to load settings: {e}")
        return None


def run_pipeline(pipeline_setting_file,
                 pipelines=None,
                 frame_start=None,
                 frame_end=None,
                 frame_skip=None,
                 frames=None,
                 out_performance_log=True,
                 gpu_mode=False,
                 verbose=False,
                 plot=False,
                 transfer_targets=None):
    """
    Execute the specified stroke transfer pipeline stages.

    This function acts as the main entry point for running selected stages of the pipeline.
    It supports grouped execution via pipeline groups.

    Args:
        pipeline_setting_file (str): Path to the JSON pipeline setting file.
        pipelines (list[str], optional): List of pipeline stage names or group names to run.
            If None, all stages are executed.
        frame_start (int, optional): Starting frame index to override settings.
        frame_end (int, optional): Ending frame index to override settings.
        frame_skip (int, optional): Frame interval to override settings.
        frames (list[int], optional): Explicit list of frames to process.
        out_performance_log (bool): Whether to log performance metrics.
        gpu_mode (bool): Enable GPU acceleration for supported stages.
        verbose (bool): Enable verbose (debug) logging.
        plot (bool): Enable plotting or visualization outputs.
        transfer_targets (list[str], optional): List of attributes to transfer (e.g., ["orientation", "color", "width", "length"]).
    """
    pipelines = expand_pipeline_groups(pipelines)

    if verbose:
        set_level(level=logging.DEBUG)

    if pipelines is None:
        log_info(logger, "Executing all pipelines.")
    else:
        log_info(logger, f"Executing selected pipelines: {pipelines}")

    if out_performance_log:
        init_log()

    if frames is not None:
        log_info(logger, f"Overriding frames for test: {frames}")

    settings = load_pipeline_settings(pipeline_setting_file)

    if settings is None:
        return

    scene_name = settings.get("scene_name", None)

    frame_settings = settings.get("frame_settings", {})

    if frame_start is None:
        frame_start = frame_settings.get("start", 1)
    if frame_end is None:
        frame_end = frame_settings.get("end", 240)
    if frame_skip is None:
        frame_skip = frame_settings.get("skip", 1)

    learn_frames = frame_settings.get("learn_frames", [])
    learn_frames_extra = frame_settings.get("learn_frames_extra", [])
    interpolation_frames = frame_settings.get("interpolation_frames", [])

    resolution = settings.get("resolution", None)

    exemplar_file_templates = settings.get("exemplar_file_templates")
    raw_feature_file_templates = settings.get("raw_feature_file_templates")
    feature_file_templates = settings.get("feature_file_templates")

    basis_file_templates = settings.get("basis_file_templates")
    basis_smooth_file_templates = settings.get("basis_smooth_file_templates")

    model_files = settings.get("model_files")

    plot_file_templates = settings.get("plot_file_templates")
    video_tone_mapping_rendering_file = settings.get("video_tone_mapping_rendering_file")

    for_plot = settings.get("for_plot")

    stroke_file_templates = settings.get("stroke_file_templates")
    video_stroke_files = settings.get("video_stroke_files")

    smoothing_parameters = settings.get("smoothing_parameters", {})

    if pipelines is None or 'p2_compute_features' in pipelines:
        cwd = os.path.abspath(os.getcwd())
        os.chdir("../")
        p2_scene_name = cwd.split('/')[-1]
        p2_compute_features(scene_name=p2_scene_name,
                            json_fn=f'{p2_scene_name}/' + pipeline_setting_file,
                            frame_start=frame_start,
                            frame_end=frame_end,
                            frame_skip=frame_skip
                            )
        os.chdir(cwd)

    std_params = settings.get("standardization_parameters", {})
    if pipelines is None or 'p3_preprocess_feature_basis' in pipelines:


        p3_preprocess_feature_basis(
            raw_feature_file_templates=raw_feature_file_templates,
            feature_file_templates=feature_file_templates,
            basis_file_templates=basis_file_templates,
            basis_smooth_file_templates=basis_smooth_file_templates,
            T_infinity=std_params.get("T_infinity", 1.5),
            T_1=std_params.get("T_1", 0.6),
            q=std_params.get("q", 1.0),
            frame_start=frame_start,
            frame_end=frame_end,
            frame_skip=frame_skip,
            learn_frames=learn_frames,
            run_standardization=True,
            frames=frames,
            resolution=settings.get("resolution", None),
            t_obj=5.0,
            use_baseline_features=False,
            plot_feature_file_template=plot_file_templates.get("features", None),
            plot=plot,
            transmittance_file_template=for_plot["transmittance_file_template"],
            lambda_spatial=smoothing_parameters["lambda_spatial"],
            lambda_temporal=smoothing_parameters["lambda_temporal"],
            plot_basis_file_template=plot_file_templates.get("basis_smooth", None),
            intensity_l_file_template=for_plot["intensity_l_file_template"],
            plot_tone_mapping_rendering_file_template=plot_file_templates.get("rendering", None),
            video_tone_mapping_rendering_file=video_tone_mapping_rendering_file
        )

    if pipelines is not None and 'aux_exemplar_frame_estimation' in pipelines:
        est_params = settings.get("exemplar_frame_estimation_parameters", {})

        aux_exemplar_frame_estimation(
            scene_name,
            feature_file_templates,
            frame_start=est_params["start"],
            frame_end=est_params["end"],
            frame_skip=est_params["skip"],
            a=est_params["a"],
            max_num_exemplars=est_params["max_num_exemplars"],
            num_fit_GMM=est_params["num_fit_GMM"],
            resolution=est_params["resolution"],
            parallel=est_params["parallel"],
            out_log_file=est_params["out_log_file"],
            out_vis_dir=est_params["plot_dir"],
            plot=plot
        )

    if pipelines is None or 'p4_annotation_interpolation' in pipelines:
        p4_annotation_interpolation(
            exemplar_file_templates,
            learn_frames=learn_frames,
            interpolation_frames=[],
            plot=plot
        )

    if pipelines is None or 'p5_regression' in pipelines:
        regression_parameters = settings.get("regression_parameters", {})
        p5_regression(
            exemplar_file_templates=exemplar_file_templates,
            feature_file_templates=feature_file_templates,
            basis_smooth_file_templates=basis_smooth_file_templates,
            out_model_files=model_files,
            learn_frames=learn_frames,
            num_samples=regression_parameters["color_model_num_samples"],
            resolution=resolution,
            use_faiss=True, gpu_mode=gpu_mode,
            transmittance_file_template=for_plot["transmittance_file_template"],
            intensity_l_file_template=for_plot["intensity_l_file_template"],
            plot_dir=plot_file_templates["regression_dir"],
            plot=plot,
            transfer_targets=transfer_targets
        )

    relocation_parameters = settings.get("relocation_parameters", {})

    if pipelines is None or 'p6_relocation' in pipelines:
        if relocation_parameters["use_relocate"]:
            p6_relocation(
                relocator_setting_file=relocation_parameters["relocator_setting_file"],
                r_scale_min=relocation_parameters["r_scale_min"],
                r_scale_max=relocation_parameters["r_scale_max"],
                r_sigma=relocation_parameters["r_sigma"],
                r_scale_space=relocation_parameters["r_scale_space"],
                r_scale_time=relocation_parameters["r_scale_time"],
                learn_frames=learn_frames
            )

    attribute_file_templates = settings.get("attribute_file_templates")
    video_attribute_files = settings.get("video_attribute_files")

    if pipelines is None or 'p7_transfer' in pipelines:
        p7_transfer(
            model_files,
            feature_file_templates,
            basis_smooth_file_templates,
            attribute_file_templates,
            video_attribute_files,
            frame_start=frame_start,
            frame_end=frame_end,
            frame_skip=frame_skip,
            resolution=resolution,
            frames=frames,
            relocator_setting_file=relocation_parameters["relocator_setting_file"],
            transmittance_file_template=for_plot["transmittance_file_template"],
            intensity_l_file_template=for_plot["intensity_l_file_template"],
            plot_file_templates=plot_file_templates,
            plot=plot,
            transfer_targets=transfer_targets
        )

    if pipelines is None or 'p8_smoothing' in pipelines:
        if transfer_targets is None or "orientation" in transfer_targets:
            p8_smoothing(
                orientation_file_template=attribute_file_templates["orientation"],
                smooth_orientation_file_template=attribute_file_templates["smooth_orientation"],
                lambda_spatial=smoothing_parameters["lambda_spatial"],
                lambda_temporal=smoothing_parameters["lambda_temporal"],
                frame_start=frame_start,
                frame_end=frame_end,
                frame_skip=frame_skip,
                frames=frames,
                transmittance_file_template=for_plot["transmittance_file_template"],
                intensity_l_file_template=for_plot["intensity_l_file_template"],
                resolution=resolution,
                out_video_file=video_attribute_files["smooth_orientation"],
                plot_file_templates=plot_file_templates,
                plot=plot
            )

    region_label_mode = 'region_label_parameters' in settings
    region_label_parameters = settings.get("region_label_parameters", {})

    if region_label_mode and (pipelines is None or 'aux_transfer_region_labels' in pipelines):
        aux_transfer_region_labels(
            color_file_template=attribute_file_templates["color"],
            out_region_label_file_template=region_label_parameters["region_label_file_template"],
            frames=frames,
            num_clusters=region_label_parameters["num_clusters"]
        )

    stroke_rendering_parameters = settings.get("stroke_rendering_parameters", {})
    angular_random_offset_deg = stroke_rendering_parameters["angular_random_offset_deg"]
    random_offset_factor = stroke_rendering_parameters["random_offset_factor"]
    length_factor = stroke_rendering_parameters["length_factor"]
    width_factor = stroke_rendering_parameters["width_factor"]
    length_random_factor_relative = stroke_rendering_parameters["length_random_factor_relative"]
    width_random_factor_relative = stroke_rendering_parameters["width_random_factor_relative"]
    num_textures = stroke_rendering_parameters["num_textures"]
    texture_length_mipmap_level = stroke_rendering_parameters["texture_length_mipmap_level"]
    stroke_step_length = stroke_rendering_parameters["stroke_step_length"]
    stroke_step_length_accuracy = stroke_rendering_parameters["stroke_step_length_accuracy"]
    consecutive_failure_max = stroke_rendering_parameters["consecutive_failure_max"]
    sort_type = stroke_rendering_parameters["sort_type"]
    clip_with_undercoat_alpha = stroke_rendering_parameters["clip_with_undercoat_alpha"]
    texture_filename = stroke_rendering_parameters["texture_filename"]
    texture_for_active_set_for_new_stroke_filename = stroke_rendering_parameters[
        "texture_for_active_set_for_new_stroke_filename"]
    texture_for_active_set_for_existing_stroke_filename = stroke_rendering_parameters[
        "texture_for_active_set_for_existing_stroke_filename"]
    rendering_width = stroke_rendering_parameters["rendering_width"]
    rendering_height = stroke_rendering_parameters["rendering_height"]

    if pipelines is None or 'p9_gen_strokes' in pipelines:
        if region_label_mode:
            region_label_file_template = region_label_parameters["region_label_file_template"]
        else:
            region_label_file_template = None
        p9_gen_strokes(
            attribute_file_templates=attribute_file_templates,
            out_stroke_file_templates=stroke_file_templates,
            frame_start=frame_start,
            frame_end=frame_end,
            frame_skip=frame_skip,
            velocity_u_file_template=raw_feature_file_templates["apparent_relative_velocity_u"],
            velocity_v_file_template=raw_feature_file_templates["apparent_relative_velocity_v"],
            angular_random_offset_deg=angular_random_offset_deg,
            random_offset_factor=random_offset_factor,
            length_factor=length_factor,
            width_factor=width_factor,
            length_random_factor_relative=length_random_factor_relative,
            width_random_factor_relative=width_random_factor_relative,
            num_textures=num_textures,
            texture_length_mipmap_level=texture_length_mipmap_level,
            stroke_step_length=stroke_step_length,
            stroke_step_length_accuracy=stroke_step_length_accuracy,
            consecutive_failure_max=consecutive_failure_max,
            sort_type=sort_type,
            clip_with_undercoat_alpha=clip_with_undercoat_alpha,
            texture_filename=texture_filename,
            texture_for_active_set_for_new_stroke_filename=texture_for_active_set_for_new_stroke_filename,
            texture_for_active_set_for_existing_stroke_filename=texture_for_active_set_for_existing_stroke_filename,
            resolution=(rendering_width, rendering_height),
            out_video_stroke_file=video_stroke_files["stroke"],
            region_label_file_template=region_label_file_template
        )

    render_strokes_parameters = settings.get("render_strokes_parameters", {})
    height_texture_filename = render_strokes_parameters.get("height_texture_filename")
    combine_height_top_factor = render_strokes_parameters.get("combine_height_top_factor")
    combine_height_additive_factor = render_strokes_parameters.get("combine_height_additive_factor")
    combine_height_additive_log_factor = render_strokes_parameters.get("combine_height_additive_log_factor")
    tex_step = render_strokes_parameters.get("tex_step")
    height_scale = render_strokes_parameters.get("height_scale")
    vz = render_strokes_parameters.get("vz")
    lx = render_strokes_parameters.get("lx")
    ly = render_strokes_parameters.get("ly")
    lz = render_strokes_parameters.get("lz")
    glossiness = render_strokes_parameters.get("glossiness")
    kd = render_strokes_parameters.get("kd")
    ks = render_strokes_parameters.get("ks")
    ka = render_strokes_parameters.get("ka")
    light_intensity = render_strokes_parameters.get("light_intensity")
    canvas_scale = render_strokes_parameters.get("canvas_scale")

    pencil_mode = 'render_strokes_pencil_parameters' in settings

    if not pencil_mode and (pipelines is None or 'p10_render_strokes' in pipelines):
        p10_render_strokes(
            frame_start=frame_start,
            frame_end=frame_end,
            frame_skip=frame_skip,
            num_textures=num_textures,
            texture_length_mipmap_level=texture_length_mipmap_level,
            color_texture_filename=texture_filename,
            height_texture_filename=height_texture_filename,
            out_final_filename_template=stroke_file_templates["final"],
            stroke_data_filename_template=stroke_file_templates["stroke_data"],
            undercoat_filename_template=attribute_file_templates["undercoat"],
            combine_height_top_factor=combine_height_top_factor,
            combine_height_additive_factor=combine_height_additive_factor,
            combine_height_additive_log_factor=combine_height_additive_log_factor,
            tex_step=tex_step,
            height_scale=height_scale,
            vz=vz,
            lx=lx,
            ly=ly,
            lz=lz,
            glossiness=glossiness,
            kd=kd,
            ks=ks,
            ka=ka,
            light_intensity=light_intensity,
            canvas_scale=canvas_scale,
            resolution=(rendering_width, rendering_height),
            out_video_final_file=video_stroke_files["final"],
        )

    if pencil_mode and (pipelines is None or 'p10_render_strokes_pencil' in pipelines):
        render_strokes_pencil_parameters = settings.get("render_strokes_pencil_parameters", {})
        pencil_factor = render_strokes_pencil_parameters.get("pencil_factor", {})
        pencil_texture_filename = render_strokes_pencil_parameters.get("pencil_texture_filename", {})
        paper_texture_filename = render_strokes_pencil_parameters.get("paper_texture_filename", {})

        p10_render_strokes_pencil(
            frame_start=frame_start,
            frame_end=frame_end,
            frame_skip=frame_skip,
            num_textures=num_textures,
            texture_length_mipmap_level=texture_length_mipmap_level,
            color_texture_filename=pencil_texture_filename,

            out_final_filename_template=stroke_file_templates["final"],
            stroke_data_filename_template=stroke_file_templates["stroke_data"],
            undercoat_filename_template=attribute_file_templates["undercoat"],
            tex_step=tex_step,
            height_scale=height_scale,
            vz=vz,
            lx=lx,
            ly=ly,
            lz=lz,
            glossiness=glossiness,
            kd=kd,
            ks=ks,
            ka=ka,
            light_intensity=light_intensity,
            canvas_scale=canvas_scale,
            resolution=(rendering_width, rendering_height),
            out_video_final_file=video_stroke_files["final"],
            pencil_factor=pencil_factor,
            paper_texture_filename=paper_texture_filename,

        )

    if not pencil_mode and (pipelines is None or 'p11_final_composite' in pipelines):
        p11_final_composite(
            final_file_template=stroke_file_templates["final"],
            out_final_comp_template=stroke_file_templates["final_comp"],
            undercoat_filename_template=attribute_file_templates["undercoat"],
            frame_start=frame_start,
            frame_end=frame_end,
            frame_skip=frame_skip,
            resolution=(rendering_width, rendering_height),
            out_video_final_comp_file=video_stroke_files["final_comp"],
        )

    if "plt_reference_rendering" in pipelines:
        plt_reference_rendering(
            raw_feature_file_templates,
            plot_file_template=plot_file_templates.get("rendering", None),
            out_video_file=video_tone_mapping_rendering_file,
            T_infinity=std_params.get("T_infinity", 1.5),
            T_1=std_params.get("T_1", 0.6),
            frame_start=frame_start,
            frame_end=frame_end,
            frame_skip=frame_skip,
            frames=frames,
            transmittance_file_template=for_plot["transmittance_file_template"],
            resolution=resolution
        )

    if out_performance_log:
        close_log()
