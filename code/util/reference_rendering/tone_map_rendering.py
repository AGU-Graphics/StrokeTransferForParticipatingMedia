# -----------------------------------------------------------------------------
# Stroke Transfer for Participating Media
# http://cg.it.aoyama.ac.jp/yonghao/sig25/abstsig25.html
#
# File: util/tone_map_rendering/tone_map_rendering.py
# Maintainer: Hideki Todo
#
# Description:
# Generating tone mapped rendering results from RGB features.
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
import os

import cv2
import numpy as np
from scipy.special import expit
from tqdm import tqdm

from util.common.feature_basis_io import load_hdf5, save_rgba
from util.image.image2video import images_to_video
from util.image.lab_util import rgb_to_lab, lab_to_rgb
from util.infra.logger import log_subsection


def find_tone_mapping_parameter(T_1=0.25, T_infinity=1.5):
    """Computes parameters for sigmoid tone mapping.

    Args:
        T_1 (float): Target tone value at x = 1.
        T_infinity (float): Maximum tone value.

    Returns:
        tuple: (L_M, theta_T) tone mapping parameters.
    """
    L_M = T_infinity
    theta_T = math.log((1 + T_1 / T_infinity) / (1 - T_1 / T_infinity))
    return L_M, theta_T


def load_rgb_hdf5(r_file, g_file, b_file):
    """Loads separate R, G, B feature files and stacks them into an RGB image.

    Args:
        r_file (str): Path to red channel file.
        g_file (str): Path to green channel file.
        b_file (str): Path to blue channel file.

    Returns:
        ndarray: Combined RGB image.
    """
    r = load_hdf5(r_file)
    g = load_hdf5(g_file)
    b = load_hdf5(b_file)

    I = np.dstack([r, g, b])
    return I


def feature_alpha(transmittance_file):
    """Computes alpha mask from transmittance values.

    Args:
        transmittance_file (str): Path to HDF5 file with transmittance data.

    Returns:
        ndarray: Alpha mask in range [0, 1].
    """
    T = load_hdf5(transmittance_file)
    A = 1.0 - T ** 5
    return A


def luminance_gray(I):
    """Computes luminance using Rec. 709 grayscale weights.

    Args:
        I (ndarray): RGB image.

    Returns:
        ndarray: 2D luminance map.
    """
    return np.einsum("ijk,k->ij", I[:, :, :3], np.array([0.2126, 0.7152, 0.0722]))


def luminance_Lab(I):
    """Computes luminance from an RGB image via Lab conversion.

    Args:
        I (ndarray): RGB image.

    Returns:
        ndarray: Normalized luminance (L*) in range [0, 1].
    """
    Lab = rgb_to_lab(I)
    return Lab[:, :, 0] / 100.0


def luminance(I):
    """Computes luminance.

    Args:
        I (ndarray): RGB image.

    Returns:
        ndarray: 2D luminance map.
    """
    return luminance_gray(I)


def change_luminance_gray(C_in, L_out):
    """Adjusts RGB image luminance by scaling channels proportionally.

    Args:
        C_in (ndarray): Input RGB image.
        L_out (ndarray): Target luminance map.

    Returns:
        ndarray: Output RGB image with new luminance.
    """
    epsilon = 1e-11
    L_in = luminance(C_in)
    return np.einsum("ijk,ij->ijk", C_in, L_out / (L_in + epsilon))


def change_luminance_Lab(C, L):
    """Changes luminance of an RGB image via Lab space manipulation.

    Args:
        C (ndarray): Input RGB image.
        L (ndarray): Target luminance values [0, 1].

    Returns:
        ndarray: Output RGB image with modified L*.
    """

    Lab = rgb_to_lab(C)
    Lab[:, :, 0] = 100.0 * L
    C_out = lab_to_rgb(Lab)
    return C_out


def change_luminance(C, L):
    """Changes luminance of an RGB image using grayscale-based scaling.

    Args:
        C (ndarray): Input RGB image.
        L (ndarray): Target luminance map.

    Returns:
        ndarray: RGB image with adjusted luminance.
    """
    return change_luminance_gray(C, L)


def tone_mapping_sigmoid(L, L_M=1.39, theta_T=1.67):
    """Applies sigmoid tone mapping to luminance values.

    Args:
        L (ndarray): Input luminance map.
        L_M (float): Maximum tone value.
        theta_T (float): Steepness parameter.

    Returns:
        ndarray: Tone-mapped luminance.
    """
    x = theta_T * L
    return L_M * (2.0 * expit(x) - 1.0)


def tone_mapping_image_gray(C, L_M=1.39, theta_T=1.67):
    """Applies tone mapping using grayscale luminance adjustment.

    Args:
        C (ndarray): Input RGB image.
        L_M (float): Maximum tone value.
        theta_T (float): Steepness parameter.

    Returns:
        ndarray: RGB image with tone-mapped grayscale luminance.
    """
    L = luminance_gray(C)
    L_out = tone_mapping_sigmoid(L, L_M, theta_T)

    C_out = change_luminance_gray(C, L_out)
    C_out = np.clip(C_out, 0.0, 1.0)
    return C_out


def tone_mapping_image_Lab(C, L_M=1.39, theta_T=1.67):
    """Applies tone mapping via Lab space luminance adjustment.

    Args:
        C (ndarray): Input RGB image.
        L_M (float): Maximum tone value.
        theta_T (float): Steepness parameter.

    Returns:
        ndarray: RGB image with tone-mapped L* component.
    """
    L = luminance_Lab(C)
    L_out = tone_mapping_sigmoid(L, L_M, theta_T)

    C_out = change_luminance_Lab(C, L_out)
    C_out = np.clip(C_out, 0.0, 1.0)
    return C_out


def tone_mapping_image(C, L_M=1.39, theta_T=1.67):
    """Applies sigmoid tone mapping.

    Args:
        C (ndarray): Input RGB image.
        L_M (float): Maximum tone value.
        theta_T (float): Steepness parameter.

    Returns:
        ndarray: Tone-mapped RGB image.
    """
    C_out = np.array(C)
    for ci in range(3):
        C_out[:, :, ci] = tone_mapping_sigmoid(C[:, :, ci], L_M, theta_T)

    C_out = np.clip(C_out, 0.0, 1.0)
    return C_out


def generate_reference_rendering_from_rgb_features(
        raw_feature_file_templates,
        out_image_file_template,
        frames,
        L_M=1.39, theta_T=1.67,
        out_video_file=None,
        frame_rate=24.0,
        raw_data=False,
        bg_color=[0.0, 1.0, 0.0],
        with_bg=True,
        transmittance_file_template=None,
        resolution=None
):
    """Generate reference rendering results from RGB features.

    Args:
        raw_feature_file_templates (dict): File templates for R, G, B input features.
        out_image_file_template (str): Output path template for images.
        frames (list[int]): List of frame indices to render.
        L_M (float): Max luminance for tone mapping.
        theta_T (float): Sigmoid steepness.
        out_video_file (str, optional): Output path for video file.
        frame_rate (float): Frame rate for video export.
        raw_data (bool): If True, skip tone mapping.
        bg_color (list[float]): RGB background color.
        with_bg (bool): Whether to apply alpha compositing.
        transmittance_file_template (str): Template for transmittance file per frame.
        resolution (tuple, optional): Resize resolution (width, height).
    """
    log_subsection("Make Rendering Images")
    for frame in tqdm(frames):
        A = feature_alpha(transmittance_file_template % frame)

        r_file = raw_feature_file_templates["intensity_r"] % frame
        g_file = raw_feature_file_templates["intensity_g"] % frame
        b_file = raw_feature_file_templates["intensity_b"] % frame

        I = load_rgb_hdf5(r_file, g_file, b_file)
        O = np.array(I)

        if not raw_data:
            O = tone_mapping_image(I, L_M=L_M, theta_T=theta_T)

        if with_bg:
            for ci in range(3):
                O[:, :, ci] = (1 - A) * bg_color[ci] + A * O[:, :, ci]

        O = np.flipud(O)
        O = np.clip(O, 0.0, 1.0)

        if resolution is not None:
            O = cv2.resize(O, resolution)

        out_file = out_image_file_template % frame
        os.makedirs(os.path.dirname(out_file), exist_ok=True)
        save_rgba(out_file, O)

    if out_video_file is not None:
        log_subsection("Make Video")
        images_to_video(out_image_file_template, out_video_file, frame_rate, frames)
