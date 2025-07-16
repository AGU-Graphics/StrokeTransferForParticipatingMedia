# -----------------------------------------------------------------------------
# Stroke Transfer for Participating Media
# http://cg.it.aoyama.ac.jp/yonghao/sig25/abstsig25.html
#
# File: util/annotation_interpolation/annotation.py
# Maintainer: Hideki Todo
#
# Description:
# Manages stroke annotation data and interpolates vector fields using RBF fitting.
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

import numpy as np
from matplotlib import pyplot as plt

from util.annotation_interpolation.interpolation import fit_rbf_from_annotations, generate_vector_field_on_grid, generate_vector_field_image
from util.annotation_interpolation.vis_utils import plot_image, load_rgba, save_annotation_plot
from util.infra.logger import getLogger, log_debug

logger = getLogger()


class Annotation:
    """Represents a single stroke annotation (polyline + width)."""

    def __init__(self, positions, width):
        """

        Args:
            positions: (n, 2) np.array data for position list.
            width: float parameter value for the width.
        """
        self.positions = positions
        self.width = width

    def __repr__(self):
        return f'Annotation(positions={self.positions}, width={self.width})'


class AnnotationSet:
    """Stores a group of annotations and computes interpolated vector field."""

    def __init__(self, annotations, exemplar_image=None, normal_image=None, alpha_mask=None):
        """

        Args:
            annotations: list of annotation data (Annotation).
            exemplar_image: (h, w, 4) exemplar image drawn by artist.
            normal_image: (h, w, 4) normal image.
            alpha_mask: (h, w) alpha mask image.
        """
        self.annotations = annotations

        self.exemplar_image = exemplar_image
        self.normal_image = normal_image

        if alpha_mask is None:
            alpha_mask = normal_image[:, :, 3]
        self.alpha_mask = alpha_mask

        if exemplar_image is not None:
            self.im_shape = exemplar_image.shape

        self.model = fit_rbf_from_annotations(annotations)

        u, L, W = generate_vector_field_image(self.model, self.exemplar_image, normal_image)

        self.u = u
        self.L = L
        self.W = W
        self.alpha_mask = alpha_mask

    def orientation_image(self):
        """ Return the interpolated orientation image. """
        return self.u

    def stroke_length(self):
        """ Return the interpolated length image. """
        return self.L

    def stroke_width(self):
        """ Return the interpolated width image. """
        return self.W

    def exemplar_image(self):
        """ Return the exemplar image. """
        return self.exemplar_image

    def plot_exemplar_image(self):
        plot_image(self.exemplar_image)

    def plot_annotations(self, width_scale=1.0):
        h, w = self.im_shape[:2]
        max_size = max(h, w)
        for annotation in self.annotations:
            P = annotation.positions

            width = max(annotation.width * max_size, 3.0)

            plt.plot(P[:, 0] * max_size, P[:, 1] * max_size, "o-", linewidth=0.5 * width * width_scale,
                     markersize=0.7 * width * width_scale)

    def plot_orientations(self, num_grids=40):
        h, w = self.im_shape[:2]
        N = self.normal_image
        P, u = generate_vector_field_on_grid(self.model, N, num_grids=num_grids)

        plt.quiver(P[:, 0], P[:, 1], u[:, 0], -u[:, 1], color=[0.05, 0.05, 0.05])


def load_annotation_file(annotation_file):
    """ Load annotation data from JSON file.

    Args:
        annotation_file: input annotation data file (.json).

    Returns:
        annotations: list of annotation data (Annotation).
    """
    with open(annotation_file, 'r') as f:
        json_data = f.read()
        data = json.loads(json_data)

    annotations = []
    version = "1.0"
    for i, annotation in enumerate(data):
        if i == 0:
            if "v" in annotation.keys():
                version = annotation["v"]
        x = np.array(annotation["x"])
        y = np.array(annotation["y"])
        positions = np.array([x, y]).T
        width = annotation["width"]

        annotations.append(Annotation(positions, width))

    log_debug(logger, f"num_annotations: {len(annotations)}")

    return annotations, version


def save_annotation_file(annotations, annotation_file):
    """ Save annotation data to JSON file.

    Args:
        annotations: list of annotation data (Annotation).
        annotation_file: output annotation data file (.json)
    """
    data = []

    for i, annotation in enumerate(annotations):
        data_i = {}
        positions = annotation.positions
        data_i["x"] = positions[:, 0].tolist()
        data_i["y"] = positions[:, 1].tolist()
        data_i["width"] = annotation.width

        if i == 0:
            data_i["v"] = "0.0"
        data.append(data_i)

    with open(annotation_file, 'w') as f:
        json.dump(data, f, indent=4)


def load_annotation_set_file(annotation_file, exemplar_file):
    """Construct AnnotationSet from file inputs."""
    style_img = load_rgba(exemplar_file)
    h, w = style_img.shape[:2]
    N = None

    A = np.ones((h, w), dtype=np.float32)

    annotations, version = load_annotation_file(annotation_file)

    return AnnotationSet(annotations, style_img, N, A)


def save_annotation_plot_file(annotation_file, exemplar_file,
                              with_exemplar=True,
                              is_white=True, with_vf=False, out_file=None,
                              white_factor=0.5
                              ):
    """Save plot of annotation set (with optional overlays)."""
    annotation_set = load_annotation_set_file(annotation_file, exemplar_file)
    save_annotation_plot(
        annotation_set,
        with_exemplar=with_exemplar,
        is_white=is_white, with_vf=with_vf, out_file=out_file,
        white_factor=white_factor
    )

