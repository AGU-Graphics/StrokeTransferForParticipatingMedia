# -----------------------------------------------------------------------------
# Stroke Transfer for Participating Media
# http://cg.it.aoyama.ac.jp/yonghao/sig25/abstsig25.html
#
# File: util/normalize/norm.py
# Maintainer: Hideki Todo
#
# Description:
# Utility functions to normalize vectors and vector fields.
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
import numpy as np


def normalize_vectors(vectors):
    """
    Normalize a batch of 2D or 3D vectors.

    Args:
        vectors (np.ndarray): Array of shape (N, D) representing N vectors of dimension D.

    Returns:
        np.ndarray: Array of normalized vectors with the same shape as input.
    """
    epsilon = 1e-10
    norms = np.sqrt(np.sum(vectors * vectors, axis=1))
    return np.einsum("ij,i->ij", vectors, 1.0 / (epsilon + norms))


def normalize_vector_image(vector_field):
    """
    Normalize each vector in a 2D image of vector fields.

    Args:
        vector_field (np.ndarray): Array of shape (H, W, D) representing a vector field.

    Returns:
        np.ndarray: Normalized vector field of the same shape.
    """
    epsilon = 1e-16
    norms = epsilon + np.sqrt(np.einsum("ijk,ijk->ij", vector_field, vector_field))
    vector_field_normalized = np.einsum("ijk, ij->ijk", vector_field, 1.0 / norms)
    return vector_field_normalized
