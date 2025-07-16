# -----------------------------------------------------------------------------
# Stroke Transfer for Participating Media
# http://cg.it.aoyama.ac.jp/yonghao/sig25/abstsig25.html
#
# File: util/region_labels/transfer_region_labels.py
# Maintainer: Hideki Todo
#
# Description:
# Generating region labels for the target scene.
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
import colorsys
import math
import os
from typing import Optional

import h5py
import numpy as np
import tqdm
from sklearn.cluster import KMeans

from util.common.feature_basis_io import load_image, save_image


def gen_random_colors(num_colors, seed=0):
    """Generates a list of visually distinct random RGB colors in HSV space.

    Args:
        num_colors (int): Number of random colors to generate.
        seed (int): Random seed for reproducibility.

    Returns:
        ndarray: Array of shape (num_colors, 3) with RGB values in [0, 1].
    """
    np.random.seed(seed)

    rancom_colors = []

    for i in range(num_colors):
        h = math.fmod(0.6 + i / num_colors, 1.0)
        s = np.random.uniform(0.5, 0.7)
        v = np.random.uniform(low=0.7, high=1.0)

        rancom_colors.append(colorsys.hsv_to_rgb(h, s, v))

    random_colors = np.array(rancom_colors)
    return random_colors


def apply_colors_to_clusters(cluster_labels, colors):
    """Maps cluster labels to RGB colors to produce a color-coded image.

    Args:
        cluster_labels (ndarray): 2D array of cluster indices.
        colors (ndarray): Array of RGB colors for each cluster index.

    Returns:
        ndarray: RGB image where each pixel is colored by its cluster.
    """
    cluster_image = colors[cluster_labels]
    return cluster_image


def fit_cluster_colors(cluster_labels, image):
    """Computes the mean RGB color of each cluster.

    Args:
        cluster_labels (ndarray): 2D array of cluster indices.
        image (ndarray): RGB image used for color averaging.

    Returns:
        ndarray: RGB color for each cluster.
    """
    h, w, cs = image.shape

    num_clusters = np.max(cluster_labels) + 1
    cluster_colors = np.zeros((num_clusters, cs))

    for label in range(num_clusters):
        mask = (cluster_labels == label)

        if mask.size == 0:
            continue

        mean_color = image[mask].mean(axis=0)

        cluster_colors[label] = mean_color

    return cluster_colors


class RegionLabel:
    """Represents region labels and their associated label colors."""
    def __init__(self, cluster_labels=None, cluster_colors=None):
        self.cluster_labels = cluster_labels
        self.cluster_colors = cluster_colors

        if cluster_labels is not None and cluster_colors is None:
            self.set_random_cluster_colors()

    def get_cluster_image(self):
        return apply_colors_to_clusters(self.cluster_labels, self.cluster_colors)

    def fit_cluster_colors(self, image):
        self.cluster_colors = fit_cluster_colors(self.cluster_labels, image)

    def set_random_cluster_colors(self):
        self.cluster_colors = gen_random_colors(np.max(self.cluster_labels) + 1)


def load_region_label(input_file):
    """Loads region labels and cluster colors from an HDF5 file.

    Args:
        input_file (str): Path to the input HDF5 file.

    Returns:
        RegionLabel: Loaded region label object.
    """
    region_label = None
    with h5py.File(input_file, mode="r") as f:
        cluster_labels = np.array(f["cluster_labels"][()])
        cluster_colors = np.array(f["cluster_colors"][()])

        region_label = RegionLabel(cluster_labels=cluster_labels,
                                     cluster_colors=cluster_colors)
    return region_label


def save_region_label(output_file, region_label: RegionLabel):
    """Saves region label data (labels and colors) to an HDF5 file.

    Args:
        output_file (str): Path to the output HDF5 file.
        region_label (RegionLabel): RegionLabel object to save.
    """
    with h5py.File(output_file, mode='w') as f:
        cluster_labels = region_label.cluster_labels
        cluster_colors = region_label.cluster_colors

        h, w = np.array(cluster_labels.shape)
        resolution = np.array([w, h])
        f.create_dataset("resolution", data=resolution, compression="gzip")
        f.create_dataset("cluster_labels", data=cluster_labels, compression="gzip")
        f.create_dataset("cluster_colors", data=cluster_colors, compression="gzip")


def load_region_label_frame(input_file_template="region_label/region_label_%03d.hdf5",
                            frame=1):
    """Loads a region label for a specific frame.

    Args:
        input_file_template (str): Template for HDF5 path with frame index.
        frame (int): Frame number to load.

    Returns:
        RegionLabel: Loaded region label for the frame.
    """
    input_file = input_file_template % frame
    return load_region_label(input_file)


def save_region_label_frame(output_file_template="region_label/region_label_%03d.hdf5",
                            frame=1,
                            region_label: Optional[RegionLabel] = None,
                            is_save_cluster_image=True):
    """Saves a region label and optionally its color image for a specific frame.

    Args:
        output_file_template (str): Template for output HDF5 file path.
        frame (int): Frame number to save.
        region_label (RegionLabel): RegionLabel object to save.
        is_save_cluster_image (bool): If True, saves a .png visualization.
    """
    output_file = output_file_template % frame
    out_dir = os.path.dirname(output_file)
    os.makedirs(out_dir, exist_ok=True)
    save_region_label(output_file, region_label)

    if is_save_cluster_image:
        cluster_image = region_label.get_cluster_image()

        output_file_base = os.path.splitext(output_file)[0]
        output_file = output_file_base + ".png"
        save_image(output_file, cluster_image)


def transfer_region_labels(
        color_file_template,
        out_region_label_file_template,
        frames,
        num_per_samples=10000,
        num_total_samples=10000,
        num_clusters=10
):
    """Generate region labels for the target scene.

    Args:
        color_file_template (str): Template for input RGB images.
        out_region_label_file_template (str): Template for output HDF5 files.
        frames (list[int]): List of frame indices to process.
        num_per_samples (int): Number of pixels sampled per frame.
        num_total_samples (int): Total number of samples used for k-means.
        num_clusters (int): Number of clusters to use in k-means.
    """
    I_samples = []
    for frame in tqdm.tqdm(frames, "Learning"):
        color_file = color_file_template % frame

        I = load_image(color_file)
        h, w = I.shape[:2]

        num_pixels = h * w
        I_pixels = I.reshape(num_pixels, -1)

        rand_ids = np.random.choice(num_pixels, size=num_per_samples, replace=False)

        I_samples_frame = I_pixels[rand_ids, :]

        I_samples.extend(I_samples_frame)

    I_samples = np.array(I_samples)

    num_samples = I_samples.shape[0]
    rand_ids = np.random.choice(num_samples, size=num_total_samples, replace=False)

    I_samples = I_samples[rand_ids, :]

    kmeans = KMeans(n_clusters=num_clusters)

    kmeans.fit(I_samples)

    for frame in tqdm.tqdm(frames, "Transfer"):
        color_file = color_file_template % frame
        I = load_image(color_file)

        h, w = I.shape[:2]

        num_pixels = h * w
        I_pixels = I.reshape(num_pixels, -1)

        cluster_labels = kmeans.predict(I_pixels)

        cluster_labels = cluster_labels.reshape(h, w)

        region_label = RegionLabel(cluster_labels=cluster_labels)

        save_region_label_frame(output_file_template=out_region_label_file_template,
                                frame=frame,
                                region_label=region_label)
