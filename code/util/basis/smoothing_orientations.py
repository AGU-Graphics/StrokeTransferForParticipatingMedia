# -----------------------------------------------------------------------------
# Stroke Transfer for Participating Media
# http://cg.it.aoyama.ac.jp/yonghao/sig25/abstsig25.html
#
# File: util/basis/smoothing_orientations.py
# Maintainer: Hideki Todo
#
# Description:
# Performs spatiotemporal smoothing of orientation vector fields over animation.
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

import matplotlib.pyplot as plt
import numpy as np
import scipy.sparse.linalg as sp
from scipy.sparse import csr_matrix
from tqdm import tqdm

from util.common.feature_basis_io import load_hdf5, save_hdf5
from util.infra.logger import getLogger, log_debug, log_subsection
from util.plot.common import get_feature_mask, plot_vector_field, get_bg_image
from util.regression_transfer.orientation import normalize_vector_image

logger = getLogger()


# ========= Data loading / saving =========
def load_orientation_sequence(frames,
                              orientation_file_template,
                              transmittance_file_template=None,
                              normalize_input_vector_field=True,
                              apply_mask=False,
                              use_random_factor=False,
                              random_factor=0.001):
    """
    Load and optionally normalize/mask a sequence of orientation vector fields.

    Args:
        frames (List[int]): List of frame indices to process.
        orientation_file_template (str): Template for orientation HDF5 file paths.
        transmittance_file_template (str or None): Template for transmittance file paths (for masking).
        normalize_input_vector_field (bool): Whether to normalize the input vector fields.
        apply_mask (bool): Whether to apply transmittance-based masking.
        use_random_factor (bool): Whether to add small random noise to the input.
        random_factor (float): Magnitude of uniform random noise to add if enabled.

    Returns:
        np.ndarray: Array of shape (T, H, W, C) containing the loaded vector fields.
    """
    orientation_sequence = []
    for frame in frames:
        orientation_file = orientation_file_template % frame

        if not os.path.exists(orientation_file):
            raise FileNotFoundError(f"File not found: {orientation_file}")

        orientation_frame = load_hdf5(orientation_file)

        if normalize_input_vector_field:
            orientation_frame = normalize_vector_image(orientation_frame)

        if apply_mask:
            h, w = orientation_frame.shape[:2]
            mask = get_feature_mask(transmittance_file_template % frame, resolution=(w, h))
            orientation_frame = np.einsum("ijk,ij->ijk", orientation_frame, mask)

        if use_random_factor:
            orientation_frame += np.random.uniform(-random_factor, random_factor, size=orientation_frame.shape)

        orientation_sequence.append(orientation_frame)

    return np.array(orientation_sequence)


def save_orientation_sequence(smoothed_data,
                              frames,
                              output_template,
                              normalize=True):
    """
    Save a sequence of smoothed orientation vector fields to HDF5 files.

    Args:
        smoothed_data (np.ndarray): Array of shape (T, H, W, C) representing the smoothed orientations.
        frames (List[int]): List of corresponding frame indices.
        output_template (str): Template path for output files (e.g., "..._%03d.h5").
        normalize (bool): Whether to normalize the vector field before saving.
    """
    log_subsection("Save Smoothed Orientations (Per Frame)")
    for i, frame in enumerate(tqdm(frames, desc="Save Smoothed Orientations")):
        out_file = output_template % frame
        out_dir = os.path.dirname(out_file)
        os.makedirs(out_dir, exist_ok=True)

        orientation_frame = smoothed_data[i]

        if normalize:
            orientation_frame = normalize_vector_image(orientation_frame)

        save_hdf5(out_file, orientation_frame)


# ========= Vector field smoothing =========
def smooth_channels(input_data,
                    lambda_spatial,
                    lambda_temporal,
                    dx, dy, dt,
                    res_x, res_y):
    """
    Apply spatiotemporal smoothing to each channel of the input vector field.

    Args:
        input_data (np.ndarray): Array of shape (T, H, W, C) representing vector fields over time.
        lambda_spatial (float): Spatial smoothing strength.
        lambda_temporal (float): Temporal smoothing strength.
        dx (float): Spatial resolution in x direction.
        dy (float): Spatial resolution in y direction.
        dt (float): Temporal resolution (time between frames).
        res_x (int): Width of the spatial resolution.
        res_y (int): Height of the spatial resolution.

    Returns:
        np.ndarray: Smoothed vector field with the same shape as input_data.
    """
    num_frames, _, _, num_channels = input_data.shape
    smoothed = np.empty_like(input_data)
    channels = ["x", "y"]  # Extendable if more than 2 channels

    for ch in range(num_channels):
        log_subsection(f"Solve Smoothing Channel: {channels[ch] if ch < len(channels) else ch}")
        smoothed[:, :, :, ch] = spacetime_smoothing(
            input_data[:, :, :, ch],
            lambda_spatial,
            lambda_temporal,
            dx, dy, dt,
            res_x, res_y,
            num_frames
        )

    return smoothed


def spacetime_smoothing(input_data, lambda_spatial, lambda_temporal, dx, dy, dt, res_x, res_y, num_frames):
    """Core solver for spatiotemporal smoothing using linear system with Laplacian regularization.

        Args:
            input_data (np.ndarray): Input array of shape (T, H, W).
            lambda_spatial (float): Spatial smoothness coefficient.
            lambda_temporal (float): Temporal smoothness coefficient.
            dx (float): Grid spacing in x.
            dy (float): Grid spacing in y.
            dt (float): Time interval between frames.
            res_x (int): Width of each frame.
            res_y (int): Height of each frame.
            num_frames (int): Number of frames (T).

        Returns:
            np.ndarray: Smoothed result of shape (T, H, W).
        """
    D_X_slots = num_frames * (res_x - 1) * res_y
    D_Y_slots = num_frames * res_x * (res_y - 1)
    D_T_slots = (num_frames - 1) * res_x * res_y

    num_D_entries = 2 * (D_X_slots + D_Y_slots + D_T_slots)
    sparse_D_row = np.zeros(num_D_entries, dtype=int)
    sparse_D_col = np.zeros(num_D_entries, dtype=int)
    sparse_D_data = np.zeros(num_D_entries, dtype=np.float32)

    indices = np.arange(num_frames * res_x * res_y).reshape(input_data.shape)
    # print( "indices: ", indices )

    indices_ones_x = indices[:, :, 1:res_x]
    # print( "indices_ones_x: ", indices_ones_x )
    indices_minusones_x = indices[:, :, 0:res_x - 1]
    # print( "indices_minusones_x: ", indices_minusones_x )

    indices_ones_y = indices[:, 1:res_y, :]
    # print( "indices_ones_y: ", indices_ones_y )
    indices_minusones_y = indices[:, 0:res_y - 1, :]
    # print( "indices_minusones_y: ", indices_minusones_y )

    indices_ones_t = indices[1:num_frames, :, :]
    # print( "indices_ones_t: ", indices_ones_t )
    indices_minusones_t = indices[0:num_frames - 1, :, :]
    # print( "indices_minusones_t: ", indices_minusones_t )

    sparse_D_col = np.concatenate(
        [indices_ones_x.reshape(-1), indices_minusones_x.reshape(-1), indices_ones_y.reshape(-1),
         indices_minusones_y.reshape(-1), indices_ones_t.reshape(-1), indices_minusones_t.reshape(-1)])
    sparse_D_row = np.concatenate(
        [np.arange(D_X_slots), np.arange(D_X_slots), D_X_slots + np.arange(D_Y_slots), D_X_slots + np.arange(D_Y_slots),
         D_X_slots + D_Y_slots + np.arange(D_T_slots), D_X_slots + D_Y_slots + np.arange(D_T_slots)])
    sparse_D_data = np.concatenate(
        [np.ones(D_X_slots), -np.ones(D_X_slots), np.ones(D_Y_slots), -np.ones(D_Y_slots), np.ones(D_T_slots),
         -np.ones(D_T_slots)])

    # print( "sparse_D_col: ", sparse_D_col )
    # print( "sparse_D_row: ", sparse_D_row )
    # print( "sparse_D_data: ", sparse_D_data )

    operator_D = csr_matrix((sparse_D_data, (sparse_D_row, sparse_D_col)),
                            shape=(D_X_slots + D_Y_slots + D_T_slots, num_frames * res_x * res_y))

    L_x = lambda_spatial * dx * dy * dt / (dx * dx)
    L_y = lambda_spatial * dx * dy * dt / (dy * dy)
    L_t = lambda_temporal * dx * dy * dt / (dt * dt)

    sparse_L_col = np.arange(D_X_slots + D_Y_slots + D_T_slots)
    sparse_L_row = np.arange(D_X_slots + D_Y_slots + D_T_slots)
    sparse_L_data = np.concatenate([L_x * np.ones(D_X_slots), L_y * np.ones(D_Y_slots), L_t * np.ones(D_T_slots)])

    operator_L = csr_matrix((sparse_L_data, (sparse_L_row, sparse_L_col)),
                            shape=(D_X_slots + D_Y_slots + D_T_slots, D_X_slots + D_Y_slots + D_T_slots))

    sparse_I_col = np.arange(num_frames * res_x * res_y)
    sparse_I_row = np.arange(num_frames * res_x * res_y)
    sparse_I_data = np.ones(num_frames * res_x * res_y) * dx * dy * dt

    operator_eye = csr_matrix((sparse_I_data, (sparse_I_row, sparse_I_col)),
                              shape=(num_frames * res_x * res_y, num_frames * res_x * res_y))

    system_matrix = operator_eye + operator_D.transpose() * operator_L * operator_D
    sparsity = system_matrix.count_nonzero() / (num_frames * res_x * res_y * num_frames * res_x * res_y)
    log_debug(logger, f"sparsity: {sparsity}")

    flat_smoothed_data, exit_code = sp.cg(system_matrix, operator_eye * input_data.reshape(-1))

    log_debug(logger, f"exit_code: {exit_code}")
    return flat_smoothed_data.reshape(input_data.shape)


# ========= Optional: visualization =========
def plot_orientation_frame(
        orientation_file_template,
        frame,
        transmittance_file_template=None,
        intensity_l_file_template=None
):
    """Plot a vector field overlayed on an optional background image.

        Args:
            orientation_file_template (str): Template path for the orientation field file.
            frame (int): Frame index to load and plot.
            transmittance_file_template (Optional[str]): Template for transmittance mask image.
            intensity_l_file_template (Optional[str]): Template for luminance background image.

        Returns:
            np.ndarray: The (H, W, 2) normalized orientation field used in the plot.
        """
    orientation_file = orientation_file_template % frame
    orientation = load_hdf5(orientation_file)
    h, w = orientation.shape[:2]
    resolution = (w, h)

    orientation = orientation[:, :, :2]
    orientation = normalize_vector_image(orientation)

    mask = None
    bg_image = None
    if transmittance_file_template is not None:
        transmittance_file = transmittance_file_template % frame
        mask = get_feature_mask(transmittance_file, resolution=resolution)

        orientation = np.einsum("ijk,ij->ijk", orientation, mask)

        if intensity_l_file_template is not None:
            intensity_l_file = intensity_l_file_template % frame
            bg_image = get_bg_image(transmittance_file, intensity_l_file, resolution=resolution)

    if bg_image is not None:
        plt.imshow(bg_image, origin="lower")
    plot_vector_field(orientation)
    plt.xticks([])
    plt.yticks([])

    return orientation


# ========= Top-level smoothing API =========
def smoothing_orientations(
        orientation_file_template,
        smooth_orientation_file_template,
        transmittance_file_template,
        lambda_spatial, lambda_temporal,
        frame_start=1,
        frame_end=173,
        frame_skip=3,
        normalize_vector_field=True,
        normalize_input_vector_field=True,
        apply_mask=False,
        use_random_factor=False,
        random_factor=0.001,
        frames=None
):
    """Top-level function to perform spatiotemporal smoothing on orientation fields.

        Args:
            orientation_file_template (str): Template path for loading orientation HDF5 files.
            smooth_orientation_file_template (str): Template path to save smoothed orientations.
            transmittance_file_template (str): Template for transmittance masks.
            lambda_spatial (float): Weight for spatial smoothing.
            lambda_temporal (float): Weight for temporal smoothing.
            frame_start (int): Start frame index (inclusive).
            frame_end (int): End frame index (inclusive).
            frame_skip (int): Frame interval to skip.
            normalize_vector_field (bool): Whether to normalize output orientations.
            normalize_input_vector_field (bool): Whether to normalize input orientations.
            apply_mask (bool): Whether to mask input orientations by transmittance.
            use_random_factor (bool): Whether to add small random noise.
            random_factor (float): Magnitude of noise if enabled.
            frames (List[int], optional): Custom frame list. Overrides frame_start~frame_end if provided.
        """
    if frames is None:
        frames = range(frame_start, frame_end + 1, frame_skip)

    A = load_orientation_sequence(
        frames,
        orientation_file_template=orientation_file_template,
        transmittance_file_template=transmittance_file_template,
        normalize_input_vector_field=normalize_input_vector_field,
        apply_mask=apply_mask,
        use_random_factor=use_random_factor,
        random_factor=random_factor
    )

    num_data, res_y, res_x, c = A.shape

    dx = 1.0 / res_x
    dy = 1.0 / res_y
    dt = (1 / 24) * frame_skip

    smoothed_A = smooth_channels(
        input_data=A,
        lambda_spatial=lambda_spatial,
        lambda_temporal=lambda_temporal,
        dx=dx, dy=dy, dt=dt,
        res_x=res_x, res_y=res_y
    )

    save_orientation_sequence(
        smoothed_data=smoothed_A,
        frames=frames,
        output_template=smooth_orientation_file_template,
        normalize=normalize_vector_field
    )
