# -----------------------------------------------------------------------------
# Stroke Transfer for Participating Media
# http://cg.it.aoyama.ac.jp/yonghao/sig25/abstsig25.html
#
# File: util/regression_transfer/orientation.py
# Maintainer: Hideki Todo
#
# Description:
# Performs orientation field regression and transfer using feature-basis models.
# Includes training, prediction, and visualization tools.
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
import matplotlib.cm as cm
import matplotlib.pyplot as plt
import tqdm
from mpl_toolkits.axes_grid1 import make_axes_locatable

from util.common.feature_basis_def import FEATURE_BASIS_SYMBOLS
from util.common.feature_basis_io import *
from util.infra.logger import getLogger, log_phase, log_debug, log_subsection, log_info
from util.model.model_io import save_model
from util.model.orientation_model import VectorFieldRegressionModel
from util.normalize.norm import normalize_vector_image
from util.pipeline.time_logger import resume_log, pause_log, log_timing
from util.plot.common import plot_vector_field, get_feature_mask, get_bg_image

logger = getLogger()

colormap = cm.plasma


def save_orientation(orientation_file, orientation):
    """
    Save the 2D orientation vector field to an HDF5 file.

    Args:
        orientation_file (str): Output file path.
        orientation (ndarray): Orientation field (H×W×4), with the last two channels ignored.
    """
    orientation[:, :, 2] = 0.0
    orientation[:, :, 3] = 1.0

    out_file = orientation_file

    if not ".h5" in out_file:
        out_file += ".h5"

    save_hdf5(out_file, orientation[:, :, :2])


def set_features(model, feature_file_templates, frame, resolution, relocator=None):
    """
    Load and set feature maps for the regression model.

    Args:
        model (VectorFieldRegressionModel): The regression model.
        feature_file_templates (dict): Dictionary of feature name to file template.
        frame (int): Frame index.
        resolution (tuple): Desired resolution (width, height).
        relocator (optional): Feature relocator object.

    Returns:
        list: List of loaded feature names.
    """
    feature_names = []
    for feature_name in feature_file_templates.keys():

        feature_file_template = feature_file_templates[feature_name]
        if feature_file_template is None:
            log_debug(logger, f"Skip: {feature_name}")
            continue

        feature_file = feature_file_template % frame
        if not os.path.exists(feature_file):
            raise FileNotFoundError(f"Not found! {feature_file}")
        else:
            feature_names.append(feature_name)

        feature = load_hdf5(feature_file)
        feature = cv2.resize(feature, resolution)

        if relocator is not None:
            feature = relocator.relocate(feature_name, feature, frame)

        model.set_feature(feature_name, feature)
    return feature_names


def set_basis(model, basis_file_templates, frame, resolution):
    """
    Load and set basis vector fields for the regression model.

    Args:
        model (VectorFieldRegressionModel): The regression model.
        basis_file_templates (dict): Dictionary of basis name to file template.
        frame (int): Frame index.
        resolution (tuple): Desired resolution (width, height).

    Returns:
        ndarray: Array of basis vector fields.
    """
    basis_array = []
    for basis_name in basis_file_templates.keys():
        basis_file_template = basis_file_templates[basis_name]
        if basis_file_template is None:
            log_debug(logger, f"Skip: {basis_name}")
            continue
        basis_file = basis_file_template % frame
        if not os.path.exists(basis_file):
            raise FileNotFoundError(f"Not found! {basis_file}")
        else:
            log_debug(logger, f"Load: {basis_file}")

        basis = load_hdf5(basis_file)
        basis = cv2.resize(basis, resolution)

        basis[:, :, :2] = normalize_vector_image(basis[:, :, :2])
        basis_array.append(basis)
        basis4 = np.ones((resolution[1], resolution[0], 4), dtype=np.float32)
        basis4[:, :, 2] = 0.0
        basis4[:, :, :2] = basis

        model.set_basis(basis_name, basis4)

    return np.array(basis_array)


def plot_regression_results(
        reference_orientation, vf_model, learn_frame, out_fig_dir,
        transmittance_file_template, intensity_l_file_template,
        resolution=(500, 500),
        exts=["png"]):
    """
    Plot and save orientation regression results.

    Args:
        reference_orientation (ndarray): Ground-truth orientation.
        vf_model (VectorFieldRegressionModel): Trained regression model.
        learn_frame (int): Frame index.
        out_fig_dir (str): Output directory.
        transmittance_file_template (str): File template for mask.
        intensity_l_file_template (str): File template for luminance image.
        resolution (tuple): Image resolution.
        exts (list): File extensions for saving.
    """
    transmittance_file = transmittance_file_template % learn_frame
    intensity_l_file = intensity_l_file_template % learn_frame

    mask = get_feature_mask(transmittance_file_template % learn_frame, resolution=resolution)

    predicted_orientation = vf_model.predict()

    reference_orientation_normalized = normalize_vector_image(reference_orientation[:, :, :2])
    predicted_orientation_normalized = normalize_vector_image(predicted_orientation[:, :, :2])

    if mask is not None:
        bg_image = get_bg_image(transmittance_file, intensity_l_file, resolution=resolution)

        reference_orientation_normalized = np.einsum("ijk,ij->ijk", reference_orientation_normalized, mask)
        predicted_orientation_normalized = np.einsum("ijk,ij->ijk", predicted_orientation_normalized, mask)

    tex_size = 22

    fig = plt.figure(figsize=(19, 5))
    plt.subplots_adjust(left=0.0, bottom=0.0, right=1.0, top=1.0, wspace=0.01, hspace=None)

    ax = plt.subplot(1, 3, 1)
    if mask is not None:
        plt.imshow(bg_image, origin="lower")
    plot_vector_field(reference_orientation_normalized)
    plt.xticks([])
    plt.yticks([])

    ax = plt.subplot(1, 3, 2)
    if mask is not None:
        plt.imshow(bg_image, origin="lower")
    plot_vector_field(predicted_orientation_normalized)
    plt.xticks([])
    plt.yticks([])

    orientation_similarity = np.einsum("ijk,ijk->ij",
                                       reference_orientation_normalized,
                                       predicted_orientation_normalized)

    ax = plt.subplot(1, 3, 3)
    aximg = plt.imshow(orientation_similarity, origin="lower", cmap=colormap,
                       vmin=-1, vmax=1)
    plt.axis("off")

    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size="7%", pad=0.1)

    fig.colorbar(aximg, cax=cax)

    for ext in exts:
        out_fig_file = os.path.join(out_fig_dir, f"regression_result_{learn_frame:03d}.{ext}")
        fig.savefig(out_fig_file, bbox_inches="tight", pad_inches=0.05, transparent=False)
    plt.close()

    fig = plt.figure(figsize=(8, 8))
    ax = plt.subplot(1, 1, 1)
    if mask is not None:
        plt.imshow(bg_image, origin="lower")
    plot_vector_field(predicted_orientation_normalized)
    plt.xticks([])
    plt.yticks([])
    out_fig_file = os.path.join(out_fig_dir, f"regression_orientation_{learn_frame:03d}.png")
    fig.savefig(out_fig_file, bbox_inches="tight", pad_inches=0.05, transparent=False)
    plt.close()


def plot_model_matrix(
        vf_model, feature_names, basis_names,
        out_fig_dir="vis/regression",
        with_title=True,
        with_value_label=False,
        exts=["png"],
        out_matrix_data=False,
        font_size=18,
):
    """
    Visualize the learned model coefficient matrix.

    Args:
        vf_model (VectorFieldRegressionModel): Trained model.
        feature_names (list): List of feature names.
        basis_names (list): List of basis names.
        out_fig_dir (str): Output directory.
        with_title (bool): Whether to add axis labels.
        with_value_label (bool): Whether to annotate values.
        exts (list): File formats to save.
        out_matrix_data (bool): Whether to save matrix as HDF5.
        font_size (int): Font size for labels.
    """
    model_matrix = vf_model.model_matrix(feature_names=feature_names, basis_names=basis_names)

    rows = model_matrix.shape[0]
    cols = model_matrix.shape[1] + 3

    fig = plt.figure(figsize=(cols, rows))

    ax = plt.subplot(1, 1, 1)
    ax_img = plt.imshow(model_matrix, cmap=colormap)
    # Show all ticks and label them with the respective list entries

    feature_titles = [FEATURE_BASIS_SYMBOLS[feature_name] for feature_name in feature_names]
    feature_titles.append("1")

    basis_titles = [FEATURE_BASIS_SYMBOLS[basis_name] for basis_name in basis_names]

    if with_title:
        ax.set_yticks(np.arange(len(basis_titles)), labels=basis_titles, fontsize=font_size)
        ax.set_xticks(np.arange(len(feature_titles)), labels=feature_titles, fontsize=font_size)
        for tick in ax.get_xticklabels():
            tick.set_verticalalignment('center')
            tick.set_y(tick.get_position()[1] - 0.03)
        for tick in ax.get_yticklabels():
            tick.set_horizontalalignment('left')
            tick.set_x(tick.get_position()[0] - 0.062)

    # Loop over data dimensions and create text annotations.
    if with_value_label:
        for i in range(len(model_matrix)):
            for j in range(len(model_matrix[0])):
                plt.text(j, i, round(model_matrix[i, j], 3), ha="center", va="center", color="w")

    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size="3%", pad=0.2)
    cbar = plt.colorbar(ax_img, cax=cax)

    cbar.ax.tick_params(labelsize=0.8 * font_size)

    for ext in exts:
        out_fig_file = os.path.join(out_fig_dir, f"model_matrix.{ext}")
        fig.savefig(out_fig_file, bbox_inches="tight", pad_inches=0.05, transparent=False)
        plt.close()

    if out_matrix_data:
        save_hdf5(os.path.join(out_fig_dir, f"model_matrix.h5"), model_matrix)


def p5_orientation_regression(
        exemplar_file_templates,
        feature_file_templates,
        basis_smooth_file_templates,
        out_model_file,
        learn_frames=[1],
        resolution=(512, 512),
        out_fig_dir=None,
        mask_file_template=None,
        transmittance_file_template=None,
        intensity_l_file_template=None,
        plot=False,
        transfer_targets=None
):
    """
    Perform orientation regression from exemplar frames.

    Args:
        exemplar_file_templates (dict): Dictionary of exemplar file templates.
        feature_file_templates (dict): Dictionary of feature file templates.
        basis_smooth_file_templates (dict): Dictionary of basis file templates.
        out_model_file (str): Path to save the model.
        learn_frames (list): Frame indices for learning.
        resolution (tuple): Target resolution.
        out_fig_dir (str): Directory for saving plots.
        mask_file_template (str): Optional mask file template.
        transmittance_file_template (str): Optional transmittance file template.
        intensity_l_file_template (str): Optional intensity file template.
        plot (bool): Whether to plot results.
        transfer_targets (list[str] or None): If specified, only run regression for selected attributes
            (e.g., ["orientation", "color", "width", "length"]).

    Returns:
        VectorFieldRegressionModel: Trained model.
    """
    if transfer_targets is not None and "orientation" not in transfer_targets:
        return

    log_phase(f"Orientation Learning")
    resume_log()

    if out_fig_dir is not None and plot:
        os.makedirs(out_fig_dir, exist_ok=True)

    AW_constraints = []
    u_constraints = []

    vf_model = VectorFieldRegressionModel(order=1)

    reference_orientation_list = []

    for frame in tqdm.tqdm(learn_frames, desc="Learning"):
        orientation_file = exemplar_file_templates["orientation"] % frame
        reference_orientation = load_hdf5(orientation_file)

        reference_orientation = cv2.resize(reference_orientation, resolution)
        if reference_orientation.shape[2] < 3:
            h, w = reference_orientation.shape[:2]
            u_ = np.zeros((h, w, 4))
            u_[:, :, :2] = reference_orientation
            u_[:, :, 3] = 1.0
            reference_orientation = u_

        reference_orientation_list.append(reference_orientation)

        A = np.einsum("ijk,ijk->ij", reference_orientation[:, :, :2], reference_orientation[:, :, :2])
        A = np.clip(A, 0, 1)

        if mask_file_template is None:
            mask_image = exemplar_file_templates["alpha"] % frame
            mask_image = load_image(mask_image)
            mask_image = cv2.resize(mask_image, (resolution[0], resolution[1]))
            mask_image = mask_image[:, :, 3]
            A *= mask_image
        else:
            mask_file = mask_file_template % frame
            mask_image = load_image(mask_file)
            mask_image = cv2.resize(mask_image, (resolution[0], resolution[1]))
            if mask_image.ndim == 3:
                mask_image = mask_image[:, :, 3]
            A *= mask_image

        set_features(vf_model, feature_file_templates, frame, resolution)
        basis_array = set_basis(vf_model, basis_smooth_file_templates, frame, resolution)

        basis_norm = np.einsum("ijkl,ijkl->ijk", basis_array, basis_array)
        min_basis_norm = np.min(basis_norm, axis=0)
        A *= min_basis_norm

        h, w = reference_orientation.shape[:2]

        AW_samples, u_samples = vf_model.constraints(reference_orientation, A)

        AW_constraints.append(AW_samples)
        u_constraints.append(u_samples)

    AW_constraints = np.vstack(AW_constraints)
    u_constraints = np.hstack(u_constraints)

    vf_model.fit_constraints(AW_constraints, u_constraints)

    log_timing("regression", "orientation regression", num_frames=len(learn_frames))
    pause_log()

    out_dir = os.path.dirname(out_model_file)
    os.makedirs(out_dir, exist_ok=True)
    save_model(vf_model, out_model_file)

    # --- vis -----

    if plot:
        log_subsection("Plot Orientation Learning Images")
        for i, reference_orientation in enumerate(reference_orientation_list):
            learn_frame = learn_frames[i]

            set_features(vf_model, feature_file_templates, learn_frame, resolution)
            set_basis(vf_model, basis_smooth_file_templates, learn_frame, resolution)
            plot_regression_results(reference_orientation, vf_model, learn_frame, out_fig_dir,
                                    transmittance_file_template=transmittance_file_template,
                                    intensity_l_file_template=intensity_l_file_template,
                                    resolution=resolution)

        # New Order
        feature_names = [feature_name for feature_name in feature_file_templates.keys() if
                         feature_file_templates[feature_name] is not None]
        basis_names = [basis_name for basis_name in basis_smooth_file_templates.keys() if
                       basis_smooth_file_templates[basis_name] is not None]
        plot_model_matrix(vf_model, feature_names, basis_names,
                          out_fig_dir=out_fig_dir)

    return vf_model


def p7_orientation_transfer(vf_model,
                            feature_file_templates,
                            basis_smooth_file_templates, frames,
                            resolution=(500, 500),
                            relocator=None,
                            out_orientation_file_template=None,
                            turn_off_relocation=True):
    """
    Transfer predicted orientation to target frames.

    Args:
        vf_model (VectorFieldRegressionModel): Trained model.
        feature_file_templates (dict): Dictionary of feature templates.
        basis_smooth_file_templates (dict): Dictionary of basis templates.
        frames (list): List of target frame indices.
        resolution (tuple): Target resolution.
        relocator (optional): Optional relocator object.
        out_orientation_file_template (str): File template for saving orientation.
        turn_off_relocation (bool): If True, disable relocation.
    """
    log_phase(f"Orientation Transfer")

    if turn_off_relocation:
        relocator = None
        log_info(logger, f"Orientation Relocation: Disabled")

    for frame in tqdm.tqdm(frames, "Transfer"):
        set_features(vf_model, feature_file_templates, frame, resolution, relocator=relocator)
        set_basis(vf_model, basis_smooth_file_templates, frame, resolution)

        model_matrix_flat = vf_model.model.coef_
        model_matrix = model_matrix_flat.reshape(len(vf_model.basis_maps.items()), -1)

        vf_model.model.coef_ = model_matrix.flatten()

        predicted_orientation = vf_model.predict()
        predicted_orientation[:, :, 2] = 0.0

        out_file = out_orientation_file_template % frame
        os.makedirs(os.path.dirname(out_file), exist_ok=True)
        save_orientation(out_file, predicted_orientation)
