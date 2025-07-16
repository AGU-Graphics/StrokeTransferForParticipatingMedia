# -----------------------------------------------------------------------------
# Stroke Transfer for Participating Media
# http://cg.it.aoyama.ac.jp/yonghao/sig25/abstsig25.html
#
# File: aux_transfer_region_labels.py
# Maintainer: Hideki Todo
#
# Description:
# (Auxiliary pipeline) Generating region labels,
# designed for complex scenes where long strokes may exceed region boundaries.
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
from util.region_labels.transfer_region_labels import transfer_region_labels


def aux_transfer_region_labels(
        color_file_template,
        out_region_label_file_template,
        frames,
        num_clusters=10
):
    """ Generate region labels for the target scene.

    Args:
        color_file_template (str): Template for input RGB images.
        out_region_label_file_template (str): Template for output HDF5 files.
        frames (list[int]): List of frame indices to process.
        num_clusters (int): Number of clusters.
    """
    transfer_region_labels(
        color_file_template,
        out_region_label_file_template=out_region_label_file_template,
        frames=frames,
        num_clusters=num_clusters
    )
