# -----------------------------------------------------------------------------
# Stroke Transfer for Participating Media
# http://cg.it.aoyama.ac.jp/yonghao/sig25/abstsig25.html
#
# File: p4_annotation_interpolation.py
# Maintainer: Hideki Todo
#
# Description:
# Interpolate orientation, length, and width from annotation curves with thickness.
# Used as input to p5_regression as interpolated stroke attribute constraints.
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
import os.path

from util.annotation_interpolation.pipeline import annotation_interpolation_from_file
from util.infra.logger import log_phase
from util.pipeline.pipeline_decorator import deco_pipeline
from util.pipeline.time_logger import resume_log, log_timing


@deco_pipeline
def p4_annotation_interpolation(
        exemplar_file_templates,
        learn_frames=[169],
        interpolation_frames=[],
        plot=False
):
    """Interpolate stroke attributes from annotations for exemplar frames.

    Args:
        exemplar_file_templates (dict): Dictionary of file templates containing:
            - 'exemplar': Path template for exemplar images.
            - 'annotation': Path template for annotation curves.
            - 'orientation': Output template for orientation map.
            - 'length': Output template for stroke length map.
            - 'width': Output template for stroke width map.
            - 'annotation_plot': Output template for debug visualization.
        learn_frames (list[int]): Frame indices used as learning samples.
        interpolation_frames (list[int]): Frame indices where interpolation is applied.
        plot (bool): If True, generate and save annotation visualization.
    """
    frames = []
    frames.extend(learn_frames)
    frames.extend(interpolation_frames)

    resume_log()

    for frame in frames:
        exemplar_file = exemplar_file_templates["exemplar"] % frame
        annotation_file = exemplar_file_templates["annotation"] % frame
        out_orientation_file = exemplar_file_templates["orientation"] % frame
        out_length_file = exemplar_file_templates["length"] % frame
        out_width_file = exemplar_file_templates["width"] % frame
        out_annotation_plot_file = exemplar_file_templates["annotation_plot"] % frame

        log_phase(f"Process Annotation: Frame {frame} ({os.path.basename(annotation_file)})")

        annotation_interpolation_from_file(exemplar_file, annotation_file,
                                           out_orientation_file,
                                           out_length_file,
                                           out_width_file,
                                           out_annotation_plot_file,
                                           plot=plot)

    log_timing("annotation interpolation", "", num_frames=len(learn_frames)+len(interpolation_frames))
