# -----------------------------------------------------------------------------
# Stroke Transfer for Participating Media
# http://cg.it.aoyama.ac.jp/yonghao/sig25/abstsig25.html
#
# File: util/plot/common.py
# Maintainer: Hideki Todo
#
# Description:
# Common utilities for visualizing masks, intensity maps, and vector fields.
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
import matplotlib.pyplot as plt
import numpy as np

from util.common.feature_basis_io import load_hdf5


def get_feature_mask(
        transmittance_file, epsilon=1e-5,
        resolution=(512, 512)):
    transmittance = load_hdf5(transmittance_file)
    if transmittance is None:
        return None

    transmittance = cv2.resize(transmittance, resolution)
    mask = np.zeros_like(transmittance)
    mask[transmittance < 1.0 - epsilon] = 1.0

    return mask


def get_bg_image(transmittance_file, intensity_l_file, t=0.1,
                 resolution=(512, 512)):
    mask = get_feature_mask(transmittance_file, resolution=resolution)

    if mask is None:
        print("mask is None")
        return None

    intensity = load_hdf5(intensity_l_file)

    if intensity is None:
        print(f"intensity_l is None: {intensity_l_file}")
        h, w = mask.shape[:2]
        return np.ones((h, w, 3))

    bg_feature = intensity
    bg_feature = cv2.resize(bg_feature, resolution)

    h, w = bg_feature.shape[:2]
    bg_image = np.zeros((h, w, 3))

    alpha = mask * t

    for ci in range(3):
        bg_image[:, :, ci] = alpha * bg_feature + (1.0 - alpha)

    bg_image = np.clip(bg_image, 0, 1)

    return bg_image


def plot_vector_field(vector_field, grid_size=50, color=(0.2, 0.2, 0.7, 1.0)):
    h, w = vector_field.shape[:2]
    xs = np.linspace(0.5, w - 0.5, w)
    ys = np.linspace(0.5, h - 0.5, h)

    X, Y = np.meshgrid(xs, ys)
    dx = w / grid_size
    dy = h / grid_size

    s = int(dx)
    plt.quiver(X[::s, ::s], Y[::s, ::s], dx * vector_field[::s, ::s, 0], dy * vector_field[::s, ::s, 1], color=color,
               angles='xy',
               scale_units='xy', scale=1)
    plt.gca().set_aspect('equal', 'box')
    plt.xlim([0, w])
    plt.ylim([0, h])


def plot_vector_field_colormap(vector_field, add_white=0.9):
    h, w = vector_field.shape[:2]
    vector_field_color = np.zeros((h, w, 4))
    vector_field_color[:, :, 0] = vector_field[:, :, 0] * 0.5 + 0.5
    vector_field_color[:, :, 1] = vector_field[:, :, 1] * 0.5 + 0.5
    vector_field_color[:, :, 3] = 1.0
    vector_field_color[:, :, :2] = add_white + (1.0 - add_white) * vector_field_color[:, :, :2]

    plt.imshow(vector_field_color, origin="lower")
