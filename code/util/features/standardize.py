# -----------------------------------------------------------------------------
# Stroke Transfer for Participating Media
# http://cg.it.aoyama.ac.jp/yonghao/sig25/abstsig25.html
#
# File: util/features/standardize.py
# Maintainer: Hideki Todo
#
# Description:
# Feature standardization utilities for the p3_preprocess_feature_basis pipeline.
# Standardizes transmittance, intensity (L*, a*, b*), velocity, curvature, ... and optionally temperature into the [-1, 1] range.
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
import shutil

import cv2
import h5py
import numpy as np
from scipy.special import expit
from sklearn.preprocessing import MinMaxScaler

from util.common.feature_basis_io import load_hdf5, save_hdf5
from util.image.lab_util import rgb_to_lab
from util.infra.logger import getLogger, log_subsection, log_debug

logger = getLogger()


def load_bounding_box(bounding_box_file):
    """Load bounding box min and max coordinates from an HDF5 file.

        Args:
            bounding_box_file (str): Path to the HDF5 file containing 'bb_min' and 'bb_max'.

        Returns:
            Tuple[np.ndarray or None, np.ndarray or None]: Tuple of bounding box min and max.
        """
    bb_min = None
    bb_max = None
    with h5py.File(bounding_box_file, mode='r') as f:
        if "bb_max" in f.keys():
            bb_max = np.array(f["bb_max"][()])
        if "bb_min" in f.keys():
            bb_min = np.array(f["bb_min"][()])
    return bb_min, bb_max


def load_features(feature_file_template, frames):
    """Load and flatten feature data across multiple frames.

        Args:
            feature_file_template (str): Template string for feature file paths (e.g., 'path/%03d.h5').
            frames (Iterable[int]): List of frame indices to load.

        Returns:
            Tuple[np.ndarray, Tuple[int, int]]: Feature matrix of shape (N, H×W) and resolution (width, height).
        """
    features = []
    resolution = None

    for frame in frames:
        feature_file = feature_file_template % frame

        feature = load_hdf5(feature_file)

        if feature is None:
            print(f"not exist!: {feature_file}")
            break

        h, w = feature.shape[:2]

        resolution = (w, h)

        features.append(feature)
    feature_data = np.array([feature.flatten() for feature in features])
    return feature_data, resolution


def save_features(feature_file_template, frames, feature_data, resolution):
    """Save feature data for each frame in HDF5 format.

        Args:
            feature_file_template (str): Output path template for each frame.
            frames (Iterable[int]): List of frame indices.
            feature_data (np.ndarray): Flattened feature data for each frame.
            resolution (Tuple[int, int]): Width and height to reshape flattened data.
        """
    for i, frame in enumerate(frames):
        feature_i = feature_data[i]
        w, h = resolution
        feature_i = feature_i.reshape(h, w)
        out_file = feature_file_template % frame
        save_hdf5(out_file, feature_i)


def copy_feature_files(
        raw_feature_file_templates,
        feature_file_templates,
        frames,
        feature_name
):
    """Copy raw feature files to target locations for a single feature.

        Args:
            raw_feature_file_templates (Dict[str, str]): Templates for raw input file paths.
            feature_file_templates (Dict[str, str]): Templates for output file paths.
            frames (Iterable[int]): List of frame indices.
            feature_name (str): Name of the feature to copy.
        """
    if raw_feature_file_templates[feature_name] is None:
        log_debug(logger, f"- skip: {feature_name}")
        return

    log_debug(logger, f"- {feature_name}")

    for frame in frames:
        raw_feature_file = raw_feature_file_templates[feature_name] % frame
        feature_file = feature_file_templates[feature_name] % frame

        os.makedirs(os.path.dirname(feature_file), exist_ok=True)
        shutil.copy2(raw_feature_file, feature_file)


def copy_feature_files_set(
        raw_feature_file_templates,
        feature_file_templates,
        frames,
        feature_names
):
    """Copy a set of raw feature files to output locations.

        Args:
            raw_feature_file_templates (Dict[str, str]): Templates for raw input file paths.
            feature_file_templates (Dict[str, str]): Templates for output file paths.
            frames (Iterable[int]): List of frame indices.
            feature_names (List[str]): List of feature names to copy.
        """
    log_subsection(f"Use Raw Features: {feature_names}")
    for feature_name in feature_names:
        copy_feature_files(
            raw_feature_file_templates,
            feature_file_templates,
            frames,
            feature_name
        )


def luminance(I):
    """Compute the luminance channel (L) from an RGB image.

        Args:
            I (np.ndarray): RGB image of shape (H, W, 3).

        Returns:
            np.ndarray: Luminance channel normalized to [0, 1].
        """
    Lab = rgb_to_lab(I)
    return Lab[:, :, 0] / 100.0


def tone_mapping_sigmoid(L, L_M=1.39, theta_T=1.67):
    """Apply sigmoid-based tone mapping to luminance.

        Args:
            L (np.ndarray): Luminance input.
            L_M (float, optional): Tone mapping scale.
            theta_T (float, optional): Sigmoid slope.

        Returns:
            np.ndarray: Tone-mapped luminance.
        """
    x = theta_T * L
    return L_M * (2.0 * expit(x) - 1.0)


def compute_IQR(x, q=1, masks=None):
    """Compute the interquartile range (IQR) of an array.

        Args:
            x (np.ndarray): Input array.
            q (float): Lower percentile.
            masks (np.ndarray or None): Optional mask.

        Returns:
            Tuple[float, float]: (min, max) values based on IQR.
        """
    q1 = q
    if masks is not None:
        x = x[masks > 0.5]
    q2 = 100 - q1
    x_max, x_min = np.percentile(x, [q2, q1])
    return x_min, x_max


def compute_IQR_abs_max(x, q=2, masks=None):
    """Compute the symmetric max value based on the absolute IQR.

        Args:
            x (np.ndarray): Input array.
            q (float): Percentile range.
            masks (np.ndarray or None): Optional mask.

        Returns:
            float: Symmetric absolute max value.
        """
    q1 = q
    if masks is None:
        x_abs = np.abs(x)
    else:
        x_abs = np.abs(x[masks > 0.5])
    q2 = 100 - q1
    v_q2, v_q1 = np.percentile(x_abs, [q2, q1])
    return v_q2


def compute_object_length(raw_feature_file_templates, frame):
    """Compute object size based on bounding box for a specific frame.

        Args:
            raw_feature_file_templates (Dict[str, str]): Template dict including 'bounding_box'.
            frame (int): Frame index.

        Returns:
            float or None: Object length (mean of bbox size), or None if unavailable.
        """
    bouding_box_file = raw_feature_file_templates["bounding_box"] % frame
    bb_min, bb_max = load_bounding_box(bouding_box_file)

    if bb_min is None:
        return None

    l_obj = np.mean(bb_max - bb_min)

    return l_obj


def feature_masks(transmittance_file_template, frames, epsilon=1e-5, resolution=None):
    """Generate binary masks from transmittance values for each frame.

        Args:
            transmittance_file_template (str): Template path to transmittance files.
            frames (Iterable[int]): Frame indices to process.
            epsilon (float, optional): Threshold for mask generation.
            resolution (Tuple[int, int] or None): Optional resize resolution.

        Returns:
            np.ndarray: Mask array of shape (N_frames, H×W).
        """
    masks = []

    for frame in frames:
        feature_image = load_hdf5(transmittance_file_template % frame)
        mask = np.zeros_like(feature_image)
        mask[feature_image < 1.0 - epsilon] = 1.0

        if resolution is not None:
            mask = cv2.resize(mask, resolution)
        masks.append(mask.flatten())

    masks = np.array(masks)

    return masks


def log_feature_range(feature_data, label=""):
    """Log the min/max value of a feature.

        Args:
            feature_data (np.ndarray): Input feature data.
            label (str, optional): Label for log output.
        """
    log_debug(logger, f"- {label} ∈ [{np.min(feature_data)}, {np.max(feature_data)}]")


def standardize_lab(r, g, b, L_M=1.39, theta_T=1.67):
    """Convert RGB channels to Lab space and standardize each channel.

        Args:
            r (np.ndarray): Red channel.
            g (np.ndarray): Green channel.
            b (np.ndarray): Blue channel.
            L_M (float): Maximum luminance scaling factor. Defaults to 1.39.
            theta_T (float): Slope parameter for sigmoid tone mapping. Defaults to 1.67.

        Returns:
            Tuple[np.ndarray, np.ndarray, np.ndarray]: Standardized L, a, b channels.
        """
    rgb = np.dstack([r, g, b]).reshape(-1, 1, 3)
    Lab = rgb_to_lab(rgb)
    Lab /= 100.0
    L = Lab[:, :, 0].reshape(r.shape)
    a = Lab[:, :, 1].reshape(r.shape)
    b = Lab[:, :, 2].reshape(r.shape)

    L_tone_mapped = tone_mapping_sigmoid(L, L_M=L_M, theta_T=theta_T)

    scaler = MinMaxScaler(feature_range=(-1, 1))
    scaler.fit(np.array([np.min(L_tone_mapped), np.max(L_tone_mapped)]).reshape(-1, 1))

    L_standardized = scaler.transform(L_tone_mapped.reshape(-1, 1)).reshape(L.shape)

    a_mean = np.mean(a)
    a_centered = a - a_mean

    b_mean = np.mean(b)
    b_centered = b - b_mean

    ab_max = np.max(np.abs(np.array([a_centered, b_centered])))

    scaler.fit(np.array([-ab_max, ab_max]).reshape(-1, 1))
    a = scaler.transform(a_centered.reshape(-1, 1)).reshape(a.shape)
    b = scaler.transform(b_centered.reshape(-1, 1)).reshape(b.shape)

    return L_standardized, a, b


def standardize_lab_baseline(L, a, b):
    """Min-max normalize Lab channels.

        Args:
            L (np.ndarray): Lightness channel.
            a (np.ndarray): A chromatic channel.
            b (np.ndarray): B chromatic channel.

        Returns:
            Tuple[np.ndarray, np.ndarray, np.ndarray]: Standardized L, a, b channels.
        """
    scaler = MinMaxScaler(feature_range=(-1, 1))
    scaler.fit(np.array([np.min(L), np.max(L)]).reshape(-1, 1))

    L_standardized = scaler.transform(L.reshape(-1, 1)).reshape(L.shape)

    a_mean = np.mean(a)
    a_ = a - a_mean

    b_mean = np.mean(b)
    b_ = b - b_mean

    ab_max = np.max(np.abs(np.array([a_, b_])))

    scaler.fit(np.array([-ab_max, ab_max]).reshape(-1, 1))
    a = scaler.transform(a_.reshape(-1, 1)).reshape(a.shape)
    b = scaler.transform(b_.reshape(-1, 1)).reshape(b.shape)

    return L_standardized, a, b


def standardize_gradient_feature(grad_I, masks=None):
    """Standardize gradient features using clipped standard deviation.

        Args:
            grad_I (np.ndarray): Gradient magnitude or image.
            masks (np.ndarray or None): Optional mask for statistics.

        Returns:
            np.ndarray: Standardized gradient feature with values in [-1, 1].
        """
    if masks is None:
        grad_I_std = np.std(grad_I)
        grad_I_mean = np.mean(grad_I)
    else:
        grad_I_std = np.std(grad_I[masks > 0.5])
        grad_I_mean = np.mean(grad_I[masks > 0.5])

    grad_I_max = grad_I_mean + 2 * grad_I_std
    grad_I = np.clip(grad_I, 0.0, grad_I_max)

    scaler = MinMaxScaler(feature_range=(-1, 1))
    scaler.fit(np.array([0, grad_I_max]).reshape(-1, 1))

    grad_I_standardized = scaler.transform(grad_I.reshape(-1, 1)).reshape(grad_I.shape)
    return grad_I_standardized


def standardize_curvatures(K, H, l_obj, q=1, clamp_value=False, masks=None, C_max=None):
    """Standardize Gaussian and mean curvatures based on IQR or given C_max.

        Args:
            K (np.ndarray): Gaussian curvature.
            H (np.ndarray): Mean curvature.
            l_obj (float): Object scale.
            q (float, optional): Percentile parameter for IQR. Defaults to 1.
            clamp_value (bool, optional): If True, clip results to [-1, 1]. Defaults to False.
            masks (np.ndarray or None): Optional mask.
            C_max (float or None): Maximum curvature normalization value. If None, computed from IQR.

        Returns:
            Tuple[np.ndarray, np.ndarray, float]: Standardized K, H, and the C_max used.
        """
    K_IQR = compute_IQR_abs_max(K, q, masks)
    H_IQR = compute_IQR_abs_max(H, q, masks)

    if C_max is None:
        C_max = max(K_IQR, H_IQR ** 2)

    K_standardized = K / C_max
    H_standardized = H / np.sqrt(C_max)

    if clamp_value:
        K_standardized = np.clip(K, -1, 1)
        H_standardized = np.clip(H, -1, 1)

    return K_standardized, H_standardized, C_max


def standardize_silhouette_distance(d_S, clamp_value=False):
    """Normalize silhouette distance values to [0, 1] or [-1, 1].

        Args:
            d_S (np.ndarray): Distance feature.
            clamp_value (bool, optional): Whether to clip to [-1, 1]. Defaults to False.

        Returns:
            np.ndarray: Normalized silhouette distance.
        """
    d_S_standardized = d_S / np.max(d_S)
    if clamp_value:
        d_S_standardized = np.clip(d_S_standardized, -1, 1)
    return d_S_standardized


def standardize_intensity_gradient_features(raw_feature_file_templates,
                                            feature_file_templates, frames, masks=None):
    """Standardize intensity gradient features using mean and std clipping.

        Args:
            raw_feature_file_templates (Dict[str, str]): Input path templates.
            feature_file_templates (Dict[str, str]): Output path templates.
            frames (Iterable[int]): Frame indices.
            masks (np.ndarray or None): Optional mask array.
        """
    feature_name = "intensity_l_gradient"
    log_subsection(f"Standardize Intensity Gradient Features: {feature_name}")

    intensity_l_gradient, resolution = load_features(raw_feature_file_templates[feature_name], frames)

    log_feature_range(intensity_l_gradient, label="raw")

    intensity_l_gradient_standardized = standardize_gradient_feature(intensity_l_gradient, masks)

    log_feature_range(intensity_l_gradient_standardized, label=f"standardized")

    save_features(feature_file_templates[feature_name], frames, intensity_l_gradient_standardized, resolution)


def standardize_lab_features(raw_feature_file_templates,
                             feature_file_templates, frames, L_M=1.39, theta_T=1.67, q=1, masks=None):
    """Convert RGB channels to Lab and standardize L, a, b components with tone mapping.

        Args:
            raw_feature_file_templates (Dict[str, str]): Input path templates.
            feature_file_templates (Dict[str, str]): Output path templates.
            frames (Iterable[int]): Frame indices.
            L_M (float): Tone mapping luminance scale.
            theta_T (float): Tone mapping slope.
            q (float): Percentile.
            masks (np.ndarray or None): Optional mask array.
        """
    log_subsection(f"Standardize Lab Features(L_M={L_M}, theta_T={theta_T})")

    rgb_names = [
        "intensity_r",
        "intensity_g",
        "intensity_b",
    ]

    r, resolution = load_features(raw_feature_file_templates[rgb_names[0]], frames)
    g, resolution = load_features(raw_feature_file_templates[rgb_names[1]], frames)
    b, resolution = load_features(raw_feature_file_templates[rgb_names[2]], frames)

    log_feature_range(r, label=rgb_names[0])
    log_feature_range(g, label=rgb_names[1])
    log_feature_range(b, label=rgb_names[2])

    L, astar, bstar = standardize_lab(r, g, b, L_M=L_M, theta_T=theta_T)

    Lab_names = [
        "intensity_l",
        "intensity_astar",
        "intensity_bstar",
    ]
    log_feature_range(L, label=Lab_names[0])
    log_feature_range(astar, label=Lab_names[1])
    log_feature_range(bstar, label=Lab_names[2])

    save_features(feature_file_templates[Lab_names[0]], frames, L, resolution)
    save_features(feature_file_templates[Lab_names[1]], frames, astar, resolution)
    save_features(feature_file_templates[Lab_names[2]], frames, bstar, resolution)


def standardize_lab_features_baseline(raw_feature_file_templates,
                                      feature_file_templates, frames, q=1, masks=None):
    """Baseline normalization of Lab features without tone mapping.

        Args:
            raw_feature_file_templates (Dict[str, str]): Input path templates.
            feature_file_templates (Dict[str, str]): Output path templates.
            frames (Iterable[int]): Frame indices.
            q (float): Percentile.
            masks (np.ndarray or None): Optional mask array.
        """
    log_subsection(f"Standardize Lab Features")

    lab_names = [
        "intensity_l",
        "intensity_astar",
        "intensity_bstar",
    ]

    L, resolution = load_features(raw_feature_file_templates[lab_names[0]], frames)
    astar, resolution = load_features(raw_feature_file_templates[lab_names[1]], frames)
    bstar, resolution = load_features(raw_feature_file_templates[lab_names[2]], frames)

    log_feature_range(L, label=lab_names[0])
    log_feature_range(astar, label=lab_names[1])
    log_feature_range(bstar, label=lab_names[2])

    L, astar, bstar = standardize_lab_baseline(L, astar, bstar)

    Lab_names = [
        "intensity_l",
        "intensity_astar",
        "intensity_bstar",
    ]
    log_feature_range(L, label=Lab_names[0])
    log_feature_range(astar, label=Lab_names[1])
    log_feature_range(bstar, label=Lab_names[2])

    save_features(feature_file_templates[Lab_names[0]], frames, L, resolution)
    save_features(feature_file_templates[Lab_names[1]], frames, astar, resolution)
    save_features(feature_file_templates[Lab_names[2]], frames, bstar, resolution)


def standardize_silhouette_distance_features(raw_feature_file_templates, feature_file_templates, frames):
    """Standardize silhouette distance features to [0, 1].

        Args:
            raw_feature_file_templates (Dict[str, str]): Input path templates.
            feature_file_templates (Dict[str, str]): Output path templates.
            frames (Iterable[int]): Frame indices.
        """
    feature_name = "silhouette_distance"
    if raw_feature_file_templates[feature_name] is None:
        log_subsection(f"Skip Silhouette Distance Features")
        return

    log_subsection(f"Standardize Silhouette Distance Features")
    feature_data, resolution = load_features(raw_feature_file_templates[feature_name], frames)

    log_feature_range(feature_data, label="raw")

    feature_data_standardized = standardize_silhouette_distance(feature_data)

    log_feature_range(feature_data_standardized, label="standardized")

    save_features(feature_file_templates[feature_name], frames, feature_data_standardized, resolution)


def standardize_curvature_features(raw_feature_file_templates, feature_file_templates, frames, q=2, masks=None,
                                   negate_mean_curvature=False, C_max=None):
    """Standardize curvature features (Gaussian and mean curvature) for given frames.

        Args:
            raw_feature_file_templates (Dict[str, str]): Template paths for raw features.
            feature_file_templates (Dict[str, str]): Template paths for output features.
            frames (Iterable[int]): Frame indices to process.
            q (float): Percentile value used for IQR.
            masks (np.ndarray or None): Optional mask array to restrict computation.
            negate_mean_curvature (bool, optional): If True, negates the mean curvature. Defaults to False.
            C_max (float or None): Optional curvature normalization factor.

        """
    if raw_feature_file_templates["gaussian_curvature"] is None or raw_feature_file_templates["mean_curvature"] is None:
        log_subsection(f"Skip Curvature Features")
        return
    log_subsection(f"Standardize Curvature Features(q={q})")

    l_obj = compute_object_length(raw_feature_file_templates, frame=frames[0])

    K, resolution = load_features(raw_feature_file_templates["gaussian_curvature"], frames)

    if len(K) == 0:
        logger.warning(f"- empty features")
        return

    H, resolution = load_features(raw_feature_file_templates["mean_curvature"], frames)

    if negate_mean_curvature:
        H = -H

    log_debug(logger, f"- l_obj: {l_obj}")

    log_feature_range(np.array([K, H]), label="raw")

    K_standardized, H_standardized, C_max = standardize_curvatures(K, H, l_obj, q, masks=masks, C_max=C_max)

    log_debug(logger, f"- C_max: {C_max}")

    log_feature_range(np.array([K_standardized, H_standardized]), label="standardized")

    save_features(feature_file_templates["gaussian_curvature"], frames, K_standardized, resolution)
    save_features(feature_file_templates["mean_curvature"], frames, H_standardized, resolution)


def standardize_velocity_features(raw_feature_file_templates, feature_file_templates, frames, q=2, t_obj=5.0,
                                  masks=None):
    """Standardize apparent relative velocity components using IQR scaling.

        Args:
            raw_feature_file_templates (Dict[str, str]): Template paths for raw features.
            feature_file_templates (Dict[str, str]): Template paths for output features.
            frames (Iterable[int]): Frame indices to process.
            q (float): Percentile value used for IQR.
            t_obj (float): Temporal scale factor.
            masks (np.ndarray or None): Optional mask array.
        """
    log_subsection(f"Standardize Velocity Features(q={q})")

    l_obj = compute_object_length(raw_feature_file_templates, frame=frames[0])

    if l_obj is None:
        l_obj = 1.0

    velocity_names = [
        "apparent_relative_velocity_u",
        "apparent_relative_velocity_v",
        "apparent_relative_velocity_norm",
    ]

    vx, resolution = load_features(raw_feature_file_templates[velocity_names[0]], frames)
    vy, resolution = load_features(raw_feature_file_templates[velocity_names[1]], frames)
    v_norm, resolution = load_features(raw_feature_file_templates[velocity_names[2]], frames)

    log_feature_range(np.array([vx, vy, v_norm]), label="raw")

    has_nonzero_velocity = np.max(np.abs(v_norm)) > 1e-9

    if has_nonzero_velocity:
        vx *= t_obj / l_obj
        vy *= t_obj / l_obj
        v_norm *= t_obj / l_obj

        vx_IQR = compute_IQR_abs_max(vx, q=q, masks=masks)
        vy_IQR = compute_IQR_abs_max(vy, q=q, masks=masks)

        v_max = max(vx_IQR, vy_IQR)

        # v_max = np.max(v_norm)

        vx /= v_max
        vy /= v_max
        v_norm /= v_max

    log_feature_range(np.array([vx, vy, v_norm]), label="standardized")

    save_features(feature_file_templates[velocity_names[0]], frames, vx, resolution)
    save_features(feature_file_templates[velocity_names[1]], frames, vy, resolution)
    save_features(feature_file_templates[velocity_names[2]], frames, v_norm, resolution)


def standardize_transmittance_features(raw_feature_file_templates, feature_file_templates, frames):
    """Standardize transmittance values to [-1, 1] using fixed range [0, 1].

        Args:
            raw_feature_file_templates (Dict[str, str]): Template paths for raw features.
            feature_file_templates (Dict[str, str]): Template paths for output features.
            frames (Iterable[int]): Frame indices to process.
        """
    feature_name = "transmittance"
    log_subsection(f"Standardize Transmittance Features")
    feature_data, resolution = load_features(raw_feature_file_templates[feature_name], frames)

    log_feature_range(feature_data, label="raw")

    scaler = MinMaxScaler(feature_range=(-1, 1))
    scaler.fit(np.array([0, 1]).reshape(-1, 1))

    feature_data_standardized = scaler.transform(feature_data.reshape(-1, 1)).reshape(feature_data.shape)

    log_feature_range(feature_data_standardized, label="standardized")

    save_features(feature_file_templates[feature_name], frames, feature_data, resolution)


def standardize_features_IQR(raw_feature_file_templates, feature_file_templates, frames, feature_name, q=0.01,
                             masks=None):
    """Standardize a generic feature using IQR-based scaling.

        Args:
            raw_feature_file_templates (Dict[str, str]): Template paths for raw features.
            feature_file_templates (Dict[str, str]): Template paths for output features.
            frames (Iterable[int]): Frame indices to process.
            feature_name (str): Name of the feature to standardize.
            q (float): Percentile value used for IQR.
            masks (np.ndarray or None): Optional mask array.

        """
    if raw_feature_file_templates[feature_name] is None:
        print(f"skip standardize: {feature_name}")
        return

    log_subsection(f"Standardize {feature_name}(q={q})")

    feature_data, resolution = load_features(raw_feature_file_templates[feature_name], frames)

    log_feature_range(feature_data, label="raw")

    feature_IQR = compute_IQR_abs_max(feature_data, q=q, masks=masks)

    scaler = MinMaxScaler(feature_range=(-1, 1))
    scaler.fit(np.array([0, feature_IQR]).reshape(-1, 1))

    feature_data_standardized = scaler.transform(feature_data.reshape(-1, 1)).reshape(feature_data.shape)

    log_feature_range(feature_data_standardized, label="standardized")

    save_features(feature_file_templates[feature_name], frames, feature_data_standardized, resolution)


def standardize_temperature_features(raw_feature_file_templates, feature_file_templates, frames, q=0.01, masks=None):
    """Standardize temperature features using IQR-based scaling.

        Args:
            raw_feature_file_templates (Dict[str, str]): Template paths for raw features.
            feature_file_templates (Dict[str, str]): Template paths for output features.
            frames (Iterable[int]): Frame indices to process.
            q (float): Percentile value used for IQR.
            masks (np.ndarray or None): Optional mask array.
        """
    feature_name = "temperature"

    if raw_feature_file_templates[feature_name] is None:
        log_subsection(f"Skip Temperature Features")
        return

    log_subsection(f"Standardize Temperature Features(q={q})")

    feature_data, resolution = load_features(raw_feature_file_templates[feature_name], frames)

    log_feature_range(feature_data, label="raw")

    feature_IQR = compute_IQR_abs_max(feature_data, q=q, masks=masks)

    scaler = MinMaxScaler(feature_range=(-1, 1))
    scaler.fit(np.array([0, feature_IQR]).reshape(-1, 1))

    feature_data_standardized = scaler.transform(feature_data.reshape(-1, 1)).reshape(feature_data.shape)

    log_feature_range(feature_data_standardized, label="standardized")

    save_features(feature_file_templates[feature_name], frames, feature_data_standardized, resolution)


def standardize_features(
        raw_feature_file_templates,
        feature_file_templates,
        L_M=1.0, theta_T=5.0, q=1,
        frames=range(1, 173, 3),
        t_obj=5.0
):
    """Run full standardization pipeline on all features with the parameters.

        Args:
            raw_feature_file_templates (Dict[str, str]): Input path templates.
            feature_file_templates (Dict[str, str]): Output path templates.
            L_M (float): Tone mapping luminance scale.
            theta_T (float): Tone mapping slope.
            q (float): Percentile for IQR.
            frames (Iterable[int]): Frame indices.
            t_obj (float): Temporal scale.
        """
    masks = feature_masks(raw_feature_file_templates["transmittance"], frames)
    standardize_transmittance_features(raw_feature_file_templates,
                                       feature_file_templates, frames)
    standardize_lab_features(raw_feature_file_templates,
                             feature_file_templates, frames, L_M=L_M, theta_T=theta_T, q=q, masks=masks)
    standardize_intensity_gradient_features(raw_feature_file_templates,
                                            feature_file_templates, frames, masks=masks)
    standardize_silhouette_distance_features(raw_feature_file_templates, feature_file_templates, frames)
    standardize_curvature_features(raw_feature_file_templates, feature_file_templates,
                                   frames, q=q, masks=masks)
    standardize_velocity_features(raw_feature_file_templates, feature_file_templates, frames, q=q, masks=masks,
                                  t_obj=t_obj)
    standardize_temperature_features(raw_feature_file_templates, feature_file_templates, frames, q=q, masks=masks)
    standardize_features_IQR(raw_feature_file_templates, feature_file_templates, frames, feature_name="mean_free_path",
                             q=q, masks=masks)
    normal_names = [
        "apparent_normal_x",
        "apparent_normal_y",
        "apparent_normal_z",
    ]
    copy_feature_files_set(raw_feature_file_templates, feature_file_templates,
                           frames=frames, feature_names=normal_names)


def standardize_features_baseline(
        raw_feature_file_templates,
        feature_file_templates,
        q=1,
        frames=range(1, 173, 3),
        t_obj=5.0,
        resolution=None
):
    """Run baseline standardization pipeline for baseline features.

        Args:
            raw_feature_file_templates (Dict[str, str]): Input path templates.
            feature_file_templates (Dict[str, str]): Output path templates.
            q (float): Percentile for IQR.
            frames (Iterable[int], optional): Frame indices.
            t_obj (float, optional): Temporal scale.
            resolution (Tuple[int, int] or None): Optional fixed resolution for mask resizing.
        """
    masks = feature_masks(raw_feature_file_templates["transmittance"], frames, resolution=resolution)

    standardize_lab_features_baseline(raw_feature_file_templates,
                                      feature_file_templates, frames, q=q, masks=masks)

    standardize_intensity_gradient_features(raw_feature_file_templates,
                                            feature_file_templates, frames, masks=masks)
    standardize_silhouette_distance_features(raw_feature_file_templates, feature_file_templates, frames)
    standardize_curvature_features(raw_feature_file_templates, feature_file_templates,
                                   frames, q=q, masks=masks)
    standardize_velocity_features(raw_feature_file_templates, feature_file_templates, frames, q=q, masks=masks,
                                  t_obj=t_obj)
    standardize_temperature_features(raw_feature_file_templates, feature_file_templates, frames, q=q, masks=masks)

    standardize_features_IQR(raw_feature_file_templates, feature_file_templates, frames, feature_name="mean_free_path",
                             q=q, masks=masks)

    normal_names = [
        "apparent_normal_x",
        "apparent_normal_y",
        "apparent_normal_z",
    ]

    copy_feature_files_set(raw_feature_file_templates, feature_file_templates,
                           frames=frames, feature_names=normal_names)
