# -----------------------------------------------------------------------------
# Stroke Transfer for Participating Media
# http://cg.it.aoyama.ac.jp/yonghao/sig25/abstsig25.html
#
# File: util/regression_transfer/color_width_length.py
# Maintainer: Hideki Todo
#
# Description:
# Scalar field (color, width, length) regression and transfer using exemplars and feature maps, with optional visualization.
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
import numpy as np
import tqdm

from util.common.feature_basis_io import load_hdf5, save_hdf5, load_image, load_rgba, save_rgba
from util.image.image2video import images_to_video
from util.infra.logger import getLogger, log_phase, log_debug
from util.infra.logger import log_info
from util.model.model_io import save_model
from util.model.scalar_field_model import NearestNeighborModel, ScalarFieldRegressionModel
from util.pipeline.time_logger import resume_log, pause_log, log_timing
from util.plot.width_length import plot_scalar_feature_frames
from util.relocation.feature_relocator import load_feature_dependent_relocator

logger = getLogger()

os.environ["OPENCV_IO_ENABLE_OPENEXR"] = "1"
colormap = cm.plasma


def set_features(model, feature_file_templates, frame, resolution, relocator=None):
    """
    Loads feature maps from files and sets them in the regression model.

    Args:
        model (VectorFieldRegressionModel): The regression model.
        feature_file_templates (dict): Dictionary mapping feature names to file templates.
        frame (int): Frame index to load.
        resolution (tuple): Target resolution as (width, height).
        relocator (optional): Optional relocator to apply spatial transformation.
    """
    for feature_name in feature_file_templates.keys():
        feature_file_template = feature_file_templates[feature_name]
        if feature_file_template is None:
            continue
        feature_file = feature_file_template % frame
        if not os.path.exists(feature_file):
            raise FileNotFoundError(f"Not found! {feature_file}")

        feature = load_hdf5(feature_file)
        feature = cv2.resize(feature, resolution)

        if relocator is not None:
            feature = relocator.relocate(feature_name, feature, frame)

        model.set_feature(feature_name, feature)


def save_color_images_and_video(
        color_file_template,
        frames,
        out_image_file_template=None,
        out_video_file=None,
        frame_rate=24.0):
    """
    Saves color fields to image files and optionally compiles them into a video.

    Args:
        color_file_template (str): Template path for input color .h5 files.
        frames (list[int]): Frame indices to process.
        out_image_file_template (str, optional): Template path to save output .png files.
        out_video_file (str, optional): Path to output video file.
        frame_rate (float): Frame rate for the output video.
    """
    if out_image_file_template is None:
        out_image_file_template = color_file_template.replace(".h5", ".png")

    for frame in tqdm.tqdm(frames, desc="For Undercoat"):
        I = load_hdf5(color_file_template % frame)

        save_rgba(out_image_file_template % frame, np.flipud(I))

    log_debug(logger, f"out_image_file_template: {out_image_file_template}")

    if out_video_file is not None:
        images_to_video(out_image_file_template, out_video_file, frame_rate=frame_rate, frames=frames)


def p5_scalar_field_regression(
        exemplar_file_template,
        feature_file_templates,
        learn_frames,
        resolution=(500, 500),
        num_samples=100000,
        use_alpha_mask=False,
        use_faiss=False, gpu_mode=False,
        use_linear_regression=False,
        mask_file_template=None,
        alpha_file_template=None,
        relocator_setting_file=None
):
    """
    Trains a scalar field model using exemplars and feature maps.

    Args:
        exemplar_file_template (str): Template path for exemplar images.
        feature_file_templates (dict): Mapping of feature names to file templates.
        learn_frames (list[int]): List of frame indices used for training.
        resolution (tuple): Resolution for resizing input images.
        num_samples (int): Number of samples for nearest neighbor regression.
        use_alpha_mask (bool): Whether to use alpha channel as mask.
        use_faiss (bool): Whether to use FAISS for nearest neighbor search.
        gpu_mode (bool): Whether to use GPU acceleration for FAISS.
        use_linear_regression (bool): Use linear regression instead of NN.
        mask_file_template (str, optional): Template path for mask images.
        alpha_file_template (str, optional): Template path for alpha masks.
        relocator_setting_file (str, optional): Path to relocator settings file.

    Returns:
        object: Trained scalar field model.
    """
    if use_linear_regression:
        model = ScalarFieldRegressionModel()
    else:
        model = NearestNeighborModel(num_samples=num_samples, use_faiss=use_faiss, gpu_mode=gpu_mode)

    W_u_constraints = []
    I_constraints = []

    relocator = None
    if relocator_setting_file is not None:
        relocator = load_feature_dependent_relocator(relocator_setting_file)

    for frame in tqdm.tqdm(learn_frames, desc="Learning"):
        S = load_image(exemplar_file_template % frame)
        S = cv2.resize(S, resolution)

        if mask_file_template is None:
            if alpha_file_template is not None:
                mask_file = alpha_file_template % frame
                mask_image = load_rgba(mask_file)
                mask_image = np.flipud(mask_image)
                mask_image = cv2.resize(mask_image, resolution)
                mask_image = mask_image[:, :, 3]

        else:
            mask_file = mask_file_template % frame
            mask_image = load_image(mask_file)
            mask_image = cv2.resize(mask_image, resolution)

            if mask_image.ndim == 3:
                mask_image = mask_image[:, :, 3]

        set_features(model, feature_file_templates, frame, resolution,
                     relocator=relocator)

        if use_alpha_mask or mask_file_template is not None:
            log_debug(logger, "mask is applied")
            W_u_samples, I_samples = model.constraints(S, mask_image)
        else:
            W_u_samples, I_samples = model.constraints(S, None)

        W_u_constraints.extend(W_u_samples)
        I_constraints.extend(I_samples)

    W_u_constraints = np.array(W_u_constraints)
    I_constraints = np.array(I_constraints)

    model.fit_constraints(W_u_constraints, I_constraints)
    model.clean_internal()
    return model


def p5_color_width_length_regression(
        exemplar_file_templates,
        feature_file_templates,
        learn_frames,
        resolution=(512, 512), num_samples=100000,
        use_alpha_mask=False,
        out_color_model_file=None,
        out_width_model_file=None,
        out_length_model_file=None,
        use_faiss=False, gpu_mode=False,
        use_linear_regression_width=False,
        use_linear_regression_length=False,
        mask_file_template=None,
        transfer_targets=None
):
    """
    Trains scalar field models for color, width, and length attributes.

    Args:
        exemplar_file_templates (dict): Mapping of attribute names to file templates.
        feature_file_templates (dict): Mapping of feature names to file templates.
        learn_frames (list[int]): List of frame indices for training.
        resolution (tuple): Image resolution for all inputs.
        num_samples (int): Number of samples for nearest neighbor training.
        use_alpha_mask (bool): Whether to use alpha channel as mask.
        out_color_model_file (str, optional): Path to save the color model.
        out_width_model_file (str, optional): Path to save the width model.
        out_length_model_file (str, optional): Path to save the length model.
        use_faiss (bool): Use FAISS acceleration for nearest neighbors.
        gpu_mode (bool): Use GPU for FAISS.
        use_linear_regression_width (bool): Use linear regression for width.
        use_linear_regression_length (bool): Use linear regression for length.
        mask_file_template (str, optional): Template path for mask files.
        transfer_targets (list[str] or None): If specified, only run regression for selected attributes
            (e.g., ["orientation", "color", "width", "length"]).
    """
    run_color = transfer_targets is None or "color" in transfer_targets
    run_width = transfer_targets is None or "width" in transfer_targets
    run_length = transfer_targets is None or "length" in transfer_targets

    if run_color:
        log_phase(f"Color Learning")
        exemplar_name = "exemplar"

        resume_log()
        color_model = p5_scalar_field_regression(
            exemplar_file_templates[exemplar_name],
            feature_file_templates,
            learn_frames,
            resolution=resolution,
            num_samples=num_samples,
            use_alpha_mask=use_alpha_mask,
            use_faiss=use_faiss, gpu_mode=gpu_mode,
            mask_file_template=mask_file_template,
            alpha_file_template=exemplar_file_templates["alpha"]
        )
        log_timing("regression", "color regression", num_frames=len(learn_frames))
        pause_log()
        save_model(color_model, out_color_model_file)

    if run_width:
        log_phase(f"Width Learning")
        exemplar_name = "width"

        resume_log()
        width_model = p5_scalar_field_regression(
            exemplar_file_templates[exemplar_name],
            feature_file_templates,
            learn_frames,
            resolution=resolution,
            num_samples=num_samples,
            use_alpha_mask=use_alpha_mask,
            use_linear_regression=use_linear_regression_width,
            use_faiss=use_faiss, gpu_mode=gpu_mode,
            mask_file_template=mask_file_template,
            alpha_file_template=exemplar_file_templates["alpha"])
        log_timing("regression", "width regression", num_frames=len(learn_frames))
        pause_log()

        save_model(width_model, out_width_model_file)

    if run_length:
        log_phase(f"Length Learning")
        exemplar_name = "length"

        resume_log()
        length_model = p5_scalar_field_regression(
            exemplar_file_templates[exemplar_name],
            feature_file_templates,
            learn_frames,
            resolution=resolution,
            num_samples=num_samples,
            use_alpha_mask=use_alpha_mask,
            use_linear_regression=use_linear_regression_length,
            use_faiss=use_faiss, gpu_mode=gpu_mode,
            mask_file_template=mask_file_template,
            alpha_file_template=exemplar_file_templates["alpha"])

        log_timing("regression", "length regression", num_frames=len(learn_frames))
        pause_log()

        save_model(length_model, out_length_model_file)


def p_7_scalar_field_transfer(
        model, feature_file_templates,
        frames,
        resolution=(500, 500),
        relocator=None,
        out_file_template=None):
    """
    Applies scalar field model to new frames and saves the predicted output.

    Args:
        model (object): Trained scalar field model.
        feature_file_templates (dict): Mapping of feature names to file templates.
        frames (list[int]): Frame indices to process.
        resolution (tuple): Resolution for resizing input features.
        relocator (optional): Optional relocator for feature warping.
        out_file_template (str): Template path for output .h5 files.
    """
    for frame in tqdm.tqdm(frames, "Transfer"):

        set_features(model, feature_file_templates, frame, resolution, relocator=relocator)

        h, w = list(model.feature_maps.values())[0].shape[:2]
        I_fit = model.predict((h, w, -1))

        if I_fit.shape[2] == 1:
            I_fit = I_fit.reshape(h, w)

        out_file = out_file_template % frame

        os.makedirs(os.path.dirname(out_file), exist_ok=True)
        save_hdf5(out_file, I_fit)


def p7_color_width_length_transfer(
        color_model, width_model, length_model,
        feature_file_templates,
        out_attribute_file_templates,
        out_video_attribute_files,
        frames, resolution=(500, 500),
        relocator=None,
        transmittance_file_template=None,
        with_color_bar=False,
        turn_off_width_length_relocation=True,
        out_plot_file_attributes=None,
        plot=False,
        transfer_targets=None):
    """
    Transfers color, width, and length attributes using trained models and saves outputs.

    Args:
        color_model (object): Trained model for color transfer.
        width_model (object): Trained model for width transfer.
        length_model (object): Trained model for length transfer.
        feature_file_templates (dict): Mapping of feature names to file templates.
        out_attribute_file_templates (dict): Mapping of attributes to output .h5 file templates.
        out_video_attribute_files (dict): Mapping of attributes to video file paths.
        frames (list[int]): Frame indices to process.
        resolution (tuple): Image resolution for all processing.
        relocator (optional): Feature relocator object.
        transmittance_file_template (str, optional): Template for overlaying transmittance.
        with_color_bar (bool): Whether to overlay color bars on visualizations.
        turn_off_width_length_relocation (bool): Disable relocator for width/length attributes.
        out_plot_file_attributes (dict, optional): Templates for output plot images.
        plot (bool): Whether to plot and save visualizations.
        transfer_targets (list[str] or None): If specified, only run transfer for selected attributes
            (e.g., ["orientation", "color", "width", "length"]).
    """
    run_color = transfer_targets is None or "color" in transfer_targets
    run_width = transfer_targets is None or "width" in transfer_targets
    run_length = transfer_targets is None or "length" in transfer_targets

    if run_color:
        log_phase(f"Color Transfer")

        out_color_file_template = out_attribute_file_templates["color"]

        resume_log()
        p_7_scalar_field_transfer(color_model,
                                  feature_file_templates,
                                  frames,
                                  resolution=resolution,
                                  relocator=relocator,
                                  out_file_template=out_color_file_template)
        log_timing("transfer", "color transfer", num_frames=len(frames))
        pause_log()

        save_color_images_and_video(out_color_file_template,
                                    frames,
                                    out_image_file_template=out_attribute_file_templates["undercoat"],
                                    out_video_file=out_video_attribute_files["color"])

    if run_width:
        log_phase(f"Width Transfer")

        if turn_off_width_length_relocation:
            log_info(logger, "Width Relocation: Disabled")
            lw_relocator = None
        else:
            lw_relocator = relocator

        out_width_file_template = out_attribute_file_templates["width"]

        resume_log()
        p_7_scalar_field_transfer(width_model,
                                  feature_file_templates,
                                  frames,
                                  resolution=resolution,
                                  relocator=lw_relocator,
                                  out_file_template=out_width_file_template)
        log_timing("transfer", "width transfer", num_frames=len(frames))
        pause_log()

        vmin = width_model.I_min
        vmax = width_model.I_max

        if plot:
            plot_scalar_feature_frames(
                out_width_file_template,
                output_file_template=out_plot_file_attributes["width"],
                frames=frames,
                resolution=resolution,
                out_video_file=out_video_attribute_files["width"],
                transmittance_file_template=transmittance_file_template,
                with_color_bar=with_color_bar,
                vmin=vmin,
                vmax=vmax
            )

    if run_length:
        log_phase(f"Length Transfer")
        if turn_off_width_length_relocation:
            log_info(logger, "Length Relocation: Disabled")
            lw_relocator = None
        else:
            lw_relocator = relocator

        out_length_file_template = out_attribute_file_templates["length"]

        resume_log()
        p_7_scalar_field_transfer(length_model,
                                  feature_file_templates,
                                  frames,
                                  resolution=resolution,
                                  relocator=lw_relocator,
                                  out_file_template=out_length_file_template)
        log_timing("transfer", "length transfer", num_frames=len(frames))
        pause_log()

        vmin = length_model.I_min
        vmax = length_model.I_max

        if plot:
            plot_scalar_feature_frames(
                out_length_file_template,
                output_file_template=out_plot_file_attributes["length"],
                frames=frames,
                resolution=resolution,
                out_video_file=out_video_attribute_files["length"],
                transmittance_file_template=transmittance_file_template,
                with_color_bar=with_color_bar,
                vmin=vmin,
                vmax=vmax
            )
