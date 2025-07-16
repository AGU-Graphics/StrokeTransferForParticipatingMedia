# -----------------------------------------------------------------------------
# Stroke Transfer for Participating Media
# http://cg.it.aoyama.ac.jp/yonghao/sig25/abstsig25.html
#
# File: p5_regression.py
# Maintainer: Hideki Todo
#
# Description:
# Learn regression models for stroke attributes from annotated exemplars.
# Estimates orientation, color, width, and length based on feature and basis inputs.
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
from util.pipeline.pipeline_decorator import deco_pipeline

from util.regression_transfer.color_width_length import p5_color_width_length_regression
from util.regression_transfer.orientation import p5_orientation_regression


@deco_pipeline
def p5_regression(
        exemplar_file_templates,
        feature_file_templates,
        basis_smooth_file_templates,
        out_model_files,
        learn_frames=[169],
        resolution=(512, 512),
        num_samples=100000,
        use_alpha_mask=False,
        plot_dir="plot/regression",
        use_faiss=True, gpu_mode=False,
        use_linear_regression_width=True,
        use_linear_regression_length=True,
        mask_file_template=None,
        transmittance_file_template=None,
        intensity_l_file_template=None,
        plot=False,
        transfer_targets=None
):
    """Train regression models to estimate stroke attributes from features/basis and exemplars.

    Args:
        exemplar_file_templates (dict): Templates for orientation, color, width, and length annotation data.
        feature_file_templates (dict): Templates for raw or standardized feature data.
        basis_smooth_file_templates (dict): Templates for smoothed basis fields.
        out_model_files (dict): Output paths for trained model files (keys: 'orientation', 'color', 'width', 'length').
        learn_frames (list[int]): Frame indices used for learning.
        resolution (tuple[int, int]): Feature resolution for training samples.
        num_samples (int): Number of training samples to draw for color regression.
        use_alpha_mask (bool): Whether to apply alpha mask during sampling.
        plot_dir (str): Directory to save visualization of training results.
        use_faiss (bool): Whether to use FAISS for fast nearest-neighbor search.
        gpu_mode (bool): Enable GPU acceleration for FAISS if True.
        mask_file_template (str or None): Optional mask file to restrict regression area.
        transmittance_file_template (str or None): Optional template for transmittance image input.
        intensity_l_file_template (str or None): Optional template for intensity image input.
        plot (bool): If True, save visualization of model fitting and predictions.
        transfer_targets (list[str] or None): If specified, only run regression for selected attributes
            (e.g., ["orientation", "color", "width", "length"]).
    """

    p5_orientation_regression(
        exemplar_file_templates,
        feature_file_templates,
        basis_smooth_file_templates,
        learn_frames=learn_frames,
        out_model_file=out_model_files["orientation"],
        resolution=resolution,
        out_fig_dir=plot_dir,
        mask_file_template=mask_file_template,
        transmittance_file_template=transmittance_file_template,
        intensity_l_file_template=intensity_l_file_template,
        plot=plot,
        transfer_targets=transfer_targets)

    p5_color_width_length_regression(
        exemplar_file_templates,
        feature_file_templates,
        learn_frames=learn_frames,
        resolution=resolution,
        num_samples=num_samples,
        use_alpha_mask=use_alpha_mask,
        out_color_model_file=out_model_files["color"],
        out_width_model_file=out_model_files["width"],
        out_length_model_file=out_model_files["length"],
        use_faiss=use_faiss, gpu_mode=gpu_mode,
        use_linear_regression_width=use_linear_regression_width,
        use_linear_regression_length=use_linear_regression_length,
        mask_file_template=mask_file_template,
        transfer_targets=transfer_targets
    )

