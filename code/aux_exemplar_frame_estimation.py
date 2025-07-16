# -----------------------------------------------------------------------------
# Stroke Transfer for Participating Media
# http://cg.it.aoyama.ac.jp/yonghao/sig25/abstsig25.html
#
# File: aux_exemplar_frame_estimation.py
# Maintainer: Hideki Todo
#
# Description:
# (Auxiliary pipeline) Estimate exemplar frames using GMM and BIC-based greedy selection.
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

from util.exemplar_estimation.exemplar_estimation_GMM import estimate_exemplar_frames_greedy, save_json
from util.infra.logger import log_phase, log_info, getLogger, log_debug

logger = getLogger()


def aux_exemplar_frame_estimation(
        scene_name,
        feature_file_templates,
        frame_start=6,
        frame_end=240,
        frame_skip=6,
        a=1.0 / 35.0,
        max_num_exemplars=5,
        num_fit_GMM=15,
        resolution=None,
        parallel=True,
        out_log_file=None,
        log_data=None,
        out_vis_dir=None,
        plot=False
):
    """(Auxiliary pipeline) Estimate exemplar frames using GMM and BIC-based greedy selection.

    Args:
        scene_name (str): Scene identifier to store in the result log.
        feature_file_templates (dict): Mapping of feature names to file path templates.
        frame_start (int): First frame to include in estimation.
        frame_end (int): Last frame to include in estimation.
        frame_skip (int): Frame interval (e.g., every 6 frames).
        a (float): Scaling factor for computing lambda from feature dimension.
        max_num_exemplars (int): Maximum number of exemplars to evaluate.
        num_fit_GMM (int): Number of components for GMM fitting.
        resolution (tuple[int, int] or None): Resize feature maps to this size if given.
        parallel (bool): Whether to use multithreaded computation.
        out_log_file (str or None): If provided, JSON file to save result log.
        log_data (dict or None): Dictionary to be updated with current estimation results.
        out_vis_dir (str or None): Directory for saving visualization outputs.
        plot (bool): Whether to generate and save visual plots.

    Raises:
        FileNotFoundError: If any required input feature file does not exist.

    Returns:
        None. Results are saved into a_log and optionally written to out_log_file.
    """
    log_phase("Exemplar Frame Estimation")
    log_info(logger, f"parallel: {parallel}")
    log_info(logger, f"max_num_exemplars: {max_num_exemplars}")
    log_debug(logger, f"a: {a}")
    log_debug(logger, f"num_fit_GMM: {num_fit_GMM}")

    frames = range(frame_start, frame_end + 1, frame_skip)

    for frame in frames:
        for file_template in feature_file_templates.values():
            if file_template is None:
                continue
            target_file = file_template % frame
            if not os.path.exists(target_file):
                raise FileNotFoundError(f"File not found: {target_file}")

    min_exemplars, interpolation_frame, lmbd = estimate_exemplar_frames_greedy(
        feature_file_templates,
        frames=range(frame_start, frame_end + 1, frame_skip),
        max_num_exemplars=max_num_exemplars,
        resolution=resolution,
        num_fit_GMM=num_fit_GMM,
        a=a,
        parallel=parallel,
        plot_dir=out_vis_dir,
        lmbd=None,
        stop_on_best_exemplars=True,
        plot=plot
    )

    if log_data is None:
        log_data = {}

    learn_frames_extra = []
    learn_frames_extra.extend(min_exemplars)
    learn_frames_extra.append(interpolation_frame)

    log_data[scene_name] = {
        "a": a,
        "lmbd": lmbd,
        "num_exemplars": len(min_exemplars),
        "learn_frames": min_exemplars,
        "learn_frames_extra": learn_frames_extra,
        "interpolation_frames": [interpolation_frame],
    }

    if out_log_file is not None:
        save_json(out_log_file, log_data)
