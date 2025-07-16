# -----------------------------------------------------------------------------
# Stroke Transfer for Participating Media
# http://cg.it.aoyama.ac.jp/yonghao/sig25/abstsig25.html
#
# File: util/annotation_interpolation/pipeline.py
# Maintainer: Hideki Todo
#
# Description:
# Pipeline functions for interpolating annotation data into vector fields and saving visualization plots.
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
import numpy as np
from matplotlib import pyplot as plt

from util.annotation_interpolation.annotation import load_annotation_set_file, save_annotation_plot_file
from util.common.feature_basis_io import save_hdf5
from util.infra.logger import getLogger, log_subsection, log_out_files
from util.plot.common import plot_vector_field

logger = getLogger()

COLORMAP = cm.plasma
FIGSIZE = (6, 6)
PAD = 0.05
BBOX = "tight"


def flip_and_save_hdf5(array, path, invert_y=False):
    """
    Flip the array vertically and save as HDF5.
    If invert_y is True, also flip the Y component of orientation.
    """
    array = np.flipud(array)
    if invert_y:
        array[:, :, 1] *= -1
    save_hdf5(path, array)
    return array


def save_figure(image, out_path, cmap=None):
    """
    Save a matplotlib figure for the given image using optional colormap.
    """
    fig = plt.figure(figsize=FIGSIZE)
    plt.imshow(image, cmap=cmap, origin="lower")
    plt.colorbar()
    fig.savefig(out_path, bbox_inches=BBOX, pad_inches=PAD, transparent=False)
    plt.close(fig)


def save_orientation_figure(orientation, out_path):
    """
    Save an orientation plot using a custom orientation visualization function.
    """
    fig = plt.figure(figsize=FIGSIZE)
    plot_vector_field(orientation)
    fig.savefig(out_path, bbox_inches=BBOX, pad_inches=PAD, transparent=False)
    plt.close(fig)


def save_annotation_plot_png(annotation_file, exemplar_file, out_path):
    """
    Save annotation plot with white background as PNG only (release version).
    """
    save_annotation_plot_file(annotation_file, exemplar_file,
                              out_file=out_path, white_factor=0.3)


def save_all_annotation_plots(annotation_file, exemplar_file, base_path, plot_without_white):
    """
    Save multiple versions of the annotation plot for inspection and publication.

    Parameters:
        annotation_file (str): Path to the annotation input file.
        exemplar_file (str): Path to the exemplar image file.
        base_path (str): Output file path prefix (extension will be appended).
        plot_without_white (bool): If True, also save versions without white background.

    Outputs:
        - base_path.png / .pdf: Standard plot with white background (white_factor=0.3)
        - base_path_no_white.png / .pdf: Plot without white background (white_factor=1.0)
        - base_path_annotation.png / .pdf: Annotation-only (no background, no exemplar)
    """

    # Save standard plots with white background (used in most cases)
    for ext in [".png", ".pdf"]:
        save_annotation_plot_file(annotation_file, exemplar_file,
                                  out_file=base_path + ext, white_factor=0.3)

    # Save plots with white background removed (optional)
    if plot_without_white:
        for ext in [".png", ".pdf"]:
            save_annotation_plot_file(annotation_file, exemplar_file,
                                      out_file=base_path + "_no_white" + ext, white_factor=1.0)

    # Save annotation-only plots without exemplar or white background
    for ext in [".png", ".pdf"]:
        save_annotation_plot_file(annotation_file, exemplar_file,
                                  out_file=base_path + "_annotation" + ext,
                                  is_white=False, with_exemplar=False)


def annotation_interpolation_from_file(
        exemplar_file, annotation_file,
        out_orientation_file,
        out_length_file,
        out_width_file,
        out_annotation_plot_file,
        plot=False
):
    """
    Load annotation, compute and save interpolated orientation, width, and length fields.
    Also generates corresponding visualizations.
    """

    log_subsection("Interpolate Annotation")
    annotation_set = load_annotation_set_file(annotation_file, exemplar_file)
    orientation = flip_and_save_hdf5(annotation_set.orientation_image(), out_orientation_file, invert_y=True)

    width = flip_and_save_hdf5(annotation_set.stroke_width(), out_width_file)

    length = flip_and_save_hdf5(annotation_set.stroke_length(), out_length_file)

    out_files = [out_orientation_file, out_width_file, out_length_file]
    log_out_files(logger, out_files)

    if plot:
        log_subsection("Generate Annotation Interpolation Figures")
        plot_orientation_file = out_orientation_file.replace(".h5", ".png")
        plot_width_file = out_width_file.replace(".h5", ".png")
        plot_length_file = out_length_file.replace(".h5", ".png")

        save_orientation_figure(orientation, plot_orientation_file)
        save_figure(width, plot_width_file, cmap=COLORMAP)
        save_figure(length, plot_length_file, cmap=COLORMAP)

        # Save only the standard annotation plot (PNG with white background)
        save_annotation_plot_png(annotation_file, exemplar_file, out_annotation_plot_file)

        out_files = [plot_orientation_file, plot_width_file, plot_length_file, out_annotation_plot_file]
        log_out_files(logger, out_files)
