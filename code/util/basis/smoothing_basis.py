# -----------------------------------------------------------------------------
# Stroke Transfer for Participating Media
# http://cg.it.aoyama.ac.jp/yonghao/sig25/abstsig25.html
#
# File: util/basis/smoothing_basis.py
# Maintainer: Hideki Todo
#
# Description:
# Performs spatiotemporal smoothing of basis vector fields and computes their orthogonal counterparts.
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

from util.basis.smoothing_orientations import smoothing_orientations
from util.common.feature_basis_io import load_hdf5, save_hdf5
from util.infra.logger import log_subsection, getLogger

logger = getLogger()


def rot_basis_frame(perp_file, para_file):
    """Rotate a perpendicular vector field to obtain the corresponding parallel component.

        Args:
            perp_file (str): Path to the input file containing the perpendicular vector field.
            para_file (str): Path to the output file to save the rotated (parallel) vector field.
        """
    orientation_perp = load_hdf5(perp_file)

    if orientation_perp is None:
        return

    orientation_para = np.array(orientation_perp)
    orientation_para[:, :, 0] = orientation_perp[:, :, 1]
    orientation_para[:, :, 1] = - orientation_perp[:, :, 0]

    save_hdf5(para_file, orientation_para)


def rot_basis(perp_file_template, para_file_template, frames):
    """Rotate perpendicular basis vector fields to their corresponding parallel fields over multiple frames.

        Args:
            perp_file_template (str): Template path for the perpendicular input files (e.g., "path/to/file_%03d.h5").
            para_file_template (str): Template path for the parallel output files.
            frames (List[int]): List of frame indices to process.
        """
    for frame in frames:
        perp_file = perp_file_template % frame
        para_file = para_file_template % frame
        rot_basis_frame(perp_file, para_file)


def smoothing_basis(
        basis_file_templates,
        basis_smooth_file_templates,
        transmittance_file_template,
        lambda_spatial=1.0 / 10000.0,
        lambda_temporal=1.0 / 100.0,
        frame_start=1,
        frame_end=172,
        frame_skip=3,
        run_smoothing=True,
        frames=None
):
    """Apply spatiotemporal smoothing to perpendicular basis vector fields and rotate them to obtain parallel fields.

        This function processes a set of predefined basis vector types (e.g., gradients, normals),
        applies optional smoothing using `smoothing_orientations`, and rotates the result to compute the corresponding parallel basis fields.

        Args:
            basis_file_templates (dict): Mapping from basis names (e.g., "intensity_gradient_perp") to input file templates.
            basis_smooth_file_templates (dict): Mapping from basis names to smoothed output file templates.
            transmittance_file_template (str): Template path for transmittance mask files used in smoothing.
            lambda_spatial (float): Spatial smoothing strength.
            lambda_temporal (float): Temporal smoothing strength.
            frame_start (int): Start frame index.
            frame_end (int): End frame index (inclusive).
            frame_skip (int): Step between frames to process.
            run_smoothing (bool): Whether to execute the smoothing step.
            frames (List[int] or None): Optional explicit list of frame indices. If None, it is generated from frame_start/end/skip.
        """
    
    if frames is None:
        frames = list(range(frame_start, frame_end + 1, frame_skip))

    basis_map = {
        "intensity_gradient_perp": "intensity_gradient_para",
        "silhouette_guided_perp": "silhouette_guided_para",
        "apparent_normal_perp": "apparent_normal_para",
        "apparent_relative_velocity_perp": "apparent_relative_velocity_para",
        "mean_free_path_gradient_perp": "mean_free_path_gradient_para"
    }

    for perp_name in basis_map.keys():
        if not perp_name in basis_file_templates.keys():
            log_subsection(f"Skip {perp_name}")
            continue

        log_subsection(f"Smoothing {perp_name}")

        orientation_file_template = basis_file_templates[perp_name]
        if orientation_file_template is None:
            log_subsection(f"Skip: Smoothing {perp_name}")
            continue
        smooth_orientation_file_template = basis_smooth_file_templates[perp_name]

        if run_smoothing:
            smoothing_orientations(orientation_file_template,
                                   smooth_orientation_file_template,
                                   transmittance_file_template,
                                   lambda_spatial, lambda_temporal,
                                   frame_start=frame_start,
                                   frame_end=frame_end,
                                   frame_skip=frame_skip,
                                   normalize_vector_field=True,
                                   normalize_input_vector_field=True,
                                   use_random_factor=False)

    for perp_name in basis_map.keys():
        if not perp_name in basis_smooth_file_templates.keys():
            continue

        para_name = basis_map[perp_name]
        perp_file_template = basis_smooth_file_templates[perp_name]

        if perp_file_template is None:
            log_subsection(f"Skip: Rotate Basis {perp_name} -> {para_name}")
            continue

        log_subsection(f"Rotate Basis {perp_name} -> {para_name}")

        para_file_template = basis_smooth_file_templates[para_name]
        rot_basis(perp_file_template, para_file_template, frames)
