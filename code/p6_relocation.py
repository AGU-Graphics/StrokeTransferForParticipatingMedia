# -----------------------------------------------------------------------------
# Stroke Transfer for Participating Media
# http://cg.it.aoyama.ac.jp/yonghao/sig25/abstsig25.html
#
# File: p6_relocation.py
# Maintainer: Hideki Todo
#
# Description:
# Construct feature-dependent relocation maps for color transfer.
# Used to shift feature references and avoid overfitting near exemplar frames.
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
from util.common.feature_basis_def import FEATURE_NAMES
from util.infra.logger import log_phase
from util.relocation.feature_relocator import prepare_feature_dependent_relocator


def p6_relocation(
        relocator_setting_file="relocate/octave_relocator.json",
        r_scale_min=0.01, r_scale_max=0.02, r_sigma=3,
        r_scale_space=100, r_scale_time=100000,
        learn_frames=[1],
):
    """ Construct feature-dependent relocation maps for color attribute transfer.

    Args:
        relocator_setting_file (str): Path to output JSON file storing relocation settings.
        r_scale_min (float): Minimum displacement scale for relocation noise.
        r_scale_max (float): Maximum displacement scale for relocation noise.
        r_sigma (float): Standard deviation used in generating noise fields.
        r_scale_space (float): Scaling factor for spatial features.
        r_scale_time (float): Scaling factor for temporal features.
        learn_frames (list[int]): List of learning frame indices to use for computing relocation settings.

    """
    log_phase(f"Prepare Relocation")
    prepare_feature_dependent_relocator(
        out_relocator_setting_file=relocator_setting_file,
        r_scale_min=r_scale_min, r_scale_max=r_scale_max, r_sigma=r_sigma,
        r_scale_space=r_scale_space, r_scale_time=r_scale_time,
        learn_frames=learn_frames,
        feature_names=FEATURE_NAMES
    )
