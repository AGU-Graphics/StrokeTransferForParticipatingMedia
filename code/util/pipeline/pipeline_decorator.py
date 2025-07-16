# -----------------------------------------------------------------------------
# Stroke Transfer for Participating Media
# http://cg.it.aoyama.ac.jp/yonghao/sig25/abstsig25.html
#
# File: util/pipeline/pipeline_decorator.py
# Maintainer: Hideki Todo
#
# Description:
# Logging execution details and utilities for executing and timing pipeline stages.
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
import time

from util.infra.logger import getLogger, log_section_start, log_section_done

logger = getLogger()


def deco_pipeline(func):
    """
    Decorator for pipeline functions to log start/end and timing information.

    This decorator automatically:
    - Extracts frame-related metadata (`frame_start`, `frame_end`, `frame_skip`, `learn_frames`)
      from kwargs to display in logs.
    - Logs the start and end of the decorated function using `log_section_start` and `log_section_done`.
    - Measures and logs execution time.

    Args:
        func (callable): Pipeline function to be decorated.

    Returns:
        callable: Wrapped function with logging and timing behavior.
    """
    def wrapper(*args, **kwargs):
        frame_start = kwargs.get("frame_start", None)
        frame_end = kwargs.get("frame_end", None)
        frame_skip = kwargs.get("frame_skip", None)
        learn_frames = kwargs.get("learn_frames", None)

        info_parts = []

        if frame_start is not None and frame_end is not None:
            skip_part = f", Skip: {frame_skip}" if frame_skip is not None else ""
            info_parts.append(f"Frames: {frame_start}-{frame_end}{skip_part}")

        if learn_frames is not None:
            info_parts.append(f"Learn Frames: {learn_frames}")

        frame_info = ", ".join(info_parts)
        label = func.__name__
        header = f"{label} ({frame_info})" if frame_info else label

        log_section_start(header)

        time_start = time.time()
        result = func(*args, **kwargs)
        time_end = time.time()

        duration = time_end - time_start
        log_section_done(label, duration)
        return result

    return wrapper


def run_stage(run_flag, name, pipeline_func, *args, **kwargs):
    """
    Conditionally runs a pipeline stage based on a boolean flag.

    Args:
        run_flag (bool): Whether to execute the pipeline stage.
        name (str): Name of the stage (for logging).
        pipeline_func (callable): Function to execute if `run_flag` is True.
        *args: Positional arguments for `pipeline_func`.
        **kwargs: Keyword arguments for `pipeline_func`.
    """
    if run_flag:
        pipeline_func(*args, **kwargs)
    else:
        print(f"Skip {name}")


def is_run_from_CLI():
    """
    Determines whether the script is being run from the command line or from PyCharm.

    Returns:
        bool: True if run from CLI, False if inside PyCharm.
    """
    if os.getenv('PYCHARM_HOSTED') == '1':
        print("Running in PyCharm.")
        return False

    return True
