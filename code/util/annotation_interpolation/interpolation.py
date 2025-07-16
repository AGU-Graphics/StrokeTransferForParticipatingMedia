# -----------------------------------------------------------------------------
# Stroke Transfer for Participating Media
# http://cg.it.aoyama.ac.jp/yonghao/sig25/abstsig25.html
#
# File: util/annotation_interpolation/interpolation.py
# Maintainer: Hideki Todo
#
# Description:
# Interpolates sparse stroke annotations to generate orientation, length, and width fields using an RBF model.
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
import cv2
import numpy as np
from scipy.interpolate import InterpolatedUnivariateSpline

from util.annotation_interpolation.rbf_model import RBFModel
from util.normalize.norm import normalize_vectors


def image_points(width, height):
    """
    Generate a (height, width, 2) array of pixel coordinates for an image.
    """
    xs = range(width)
    ys = range(height)
    X, Y = np.meshgrid(xs, ys)
    points = np.dstack((X, Y))
    return points


def grid_points(x_max, y_max, num_grids=20):
    """
    Generate evenly spaced grid points in [0, x_max] x [0, y_max].
    """
    xs = np.linspace(0, x_max, num_grids)
    ys = np.linspace(0, y_max, num_grids)

    X, Y = np.meshgrid(xs, ys)
    points = np.dstack((X, Y))
    return points


def compute_arc_parameters(curve_points):
    """
    Compute normalized arc-length parameterization for a given curve.
    """
    t = np.zeros((curve_points.shape[0], 1))

    for i in range(curve_points.shape[0] - 1):
        t[i + 1] = t[i] + np.linalg.norm(curve_points[i + 1, :] - curve_points[i, :])
    t /= t[-1]
    return t


def compute_arc_length(curve_points):
    """
    Compute total arc length of a polyline.
    """
    t = np.zeros((curve_points.shape[0], 1))

    for i in range(curve_points.shape[0] - 1):
        t[i + 1] = t[i] + np.linalg.norm(curve_points[i + 1, :] - curve_points[i, :])
    return t[-1]


def fit_curve_spline(curve_points):
    """
    Fit a univariate spline curve to 2D points.

    Returns:
        func (callable): interpolated 2D curve function.
        t (np.ndarray): arc parameters.
    """
    k = int(min(curve_points.shape[0] - 1, 3))

    t = compute_arc_parameters(curve_points)

    x = curve_points[:, 0].flatten()
    y = curve_points[:, 1].flatten()

    fx = InterpolatedUnivariateSpline(t, x, k=k)
    fy = InterpolatedUnivariateSpline(t, y, k=k)

    def func(t_new):
        x = fx(t_new)
        y = fy(t_new)
        return np.dstack((x, y)).reshape(-1, 2)

    return func, t


def compute_curve_directions(curve_points, dt=0.001, sp=0.05):
    """
    Estimate tangents (orientation vectors) and arc-lengths along a curve.

    Returns:
        points (np.ndarray): resampled points along the curve.
        orientations (np.ndarray): normalized tangent vectors.
        lengths (float): total curve length.
    """
    if len(curve_points) < 2:
        return [], [], []

    t = compute_arc_parameters(curve_points)

    f, t = fit_curve_spline(curve_points)

    try:
        f, t = fit_curve_spline(curve_points)
    except:
        print(f"Error: {len(curve_points)}")
        return [], [], []

    num_samples = 2 * curve_points.shape[0]

    t = np.linspace(0.0, 1.0, num_samples)
    points = f(t)
    lengths = compute_arc_length(points)
    t1 = np.clip(t + dt, 0, 1)
    t0 = np.clip(t - dt, 0, 1)
    orientations = f(t1) - f(t0)

    orientations = normalize_vectors(orientations)
    return points, orientations, lengths


def extract_rbf_constraints(annotations):
    """
    Extract vector field constraints (position, orientation, length, width) from annotations.

    Returns:
        points, orientations, lengths, widths: Arrays used for RBF fitting.
    """
    points = []
    orientations = []
    lengths = []
    widths = []

    for annotation in annotations:
        P = annotation.positions

        Wi = annotation.width

        V_, u_, L_ = compute_curve_directions(P)

        points.extend(V_)
        orientations.extend(u_)
        lengths.extend([L_ for i in range(V_.shape[0])])
        widths.extend([Wi for i in range(V_.shape[0])])

    points = np.array(points)

    orientations = np.array(orientations)
    lengths = np.array(lengths).reshape(-1, 1)
    widths = np.array(widths).reshape(-1, 1)

    return points, orientations, lengths, widths


def fit_rbf_from_annotations(annotations):
    """
    Fit an RBFModel from annotated stroke data.

    Returns:
        RBFModel trained on orientation, length, and width fields.
    """
    points, orientations, lengths, widths = extract_rbf_constraints(annotations)

    X = np.array(points)

    rbf_model = RBFModel(max_samples=1000, smoothness=1e-9)

    Y = np.hstack((orientations, lengths, widths))

    rbf_model.fit(X, Y)

    return rbf_model


def generate_vector_field_on_grid(model, normal_image, x_max=1.0, y_max=1.0, num_grids=20):
    """
    Evaluate the RBF-based vector field on a uniform grid for visualization.

    Returns:
        grid_positions (np.ndarray): positions of grid points (N, 2).
        vector_field_on_grid (np.ndarray): normalized vectors at grid positions (N, 2).
    """
    grid_positions = grid_points(x_max, y_max, num_grids)
    grid_positions = grid_positions.reshape(-1, 2)

    X = np.array(grid_positions)

    Y = model.transform(X)
    vector_field_on_grid = normalize_vectors(Y[:, :2])

    h, w = normal_image.shape[:2]
    grid_positions = X[:, :2]

    max_size = max(h, w)

    grid_positions[:, 0] *= max_size - 1
    grid_positions[:, 1] *= max_size - 1

    vector_field_on_grid[:, 0] *= max_size - 1
    vector_field_on_grid[:, 1] *= max_size - 1

    return grid_positions, vector_field_on_grid


def generate_vector_field_image(model, image, normal_image=None):
    """
    Generate full-resolution orientation, length, and width maps from an RBFModel.

    Returns:
        vector_field (np.ndarray): RGBA image containing vector orientation (xy) and mask.
        length_field (np.ndarray): Scalar field of stroke lengths.
        width_field (np.ndarray): Scalar field of stroke widths.
    """
    h, w = image.shape[:2]
    if normal_image is None:
        alpha_mask = np.ones((h, w))
    else:
        alpha_mask = normal_image[:, :, 3]

    P = np.float32(image_points(w, h))
    P = P.reshape(-1, 2)

    max_size = max(w, h)

    P[:, 0] /= max_size - 1
    P[:, 1] /= max_size - 1

    vector_field = np.zeros_like(image)

    x = np.array(P)
    X = x

    scaler = ImageScaler(scale=0.5)
    X_low = X.reshape(h, w, -1)
    X_low = scaler.transform(X_low)

    h_low, w_low = X_low.shape[:2]
    Y_low = model.transform(X_low.reshape(h_low * w_low, -1))
    Y = scaler.inverse_transform(Y_low.reshape(h_low, w_low, -1))
    Y = Y.reshape(h * w, -1)

    vector_field_flat = Y[:, :2]
    vector_field_flat = normalize_vectors(vector_field_flat)

    vector_field[:, :, :2] = vector_field_flat.reshape(vector_field.shape[0], vector_field.shape[1], -1)

    length_field = Y[:, 2].reshape(h, w)
    width_field = Y[:, 3].reshape(h, w)

    length_field = np.clip(length_field, 3.0 / max_size, 10.0)
    width_field = np.clip(width_field, 3.0 / max_size, 10.0)

    if alpha_mask is None:
        alpha_mask = image[:, :, 3]

    for ci in range(3):
        vector_field[:, :, ci] *= alpha_mask

    vector_field[:, :, 3] = alpha_mask
    return vector_field, length_field, width_field


class ImageScaler:
    """
    Utility class to downsample and upsample multi-channel images consistently.
    """

    def __init__(self, scale=0.25):
        self.scale = scale

    def transform(self, image_high, interpolation=cv2.INTER_LINEAR):
        """
        Downsample an image by the given scale factor.
        """
        scale = self.scale
        h, w, cs = image_high.shape
        self.X_shape = image_high.shape
        h_low = int(h * scale)
        w_low = int(w * scale)

        image_low = np.zeros((h_low, w_low, cs))
        for ci in range(cs):
            image_low[:, :, ci] = cv2.resize(image_high[:, :, ci], (w_low, h_low), interpolation=interpolation)
        return image_low

    def inverse_transform(self, image_low):
        """
        Upsample a low-resolution image back to original shape.
        """
        X_shape = self.X_shape
        h, w = X_shape[:2]
        cs = image_low.shape[2]
        image_high = np.zeros((h, w, cs))

        for ci in range(cs):
            image_high[:, :, ci] = cv2.resize(image_low[:, :, ci], (w, h))
        return image_high
