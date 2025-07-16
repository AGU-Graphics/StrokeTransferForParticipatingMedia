# -----------------------------------------------------------------------------
# Stroke Transfer for Participating Media
# http://cg.it.aoyama.ac.jp/yonghao/sig25/abstsig25.html
#
# File: util/pipeline/time_logger.py
# Maintainer: Hideki Todo
#
# Description:
# Simple performance logger for recording elapsed time and per-frame stats.
# Supports timestamped log files with pause/resume functionality.
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
from datetime import datetime

log_file = None
last_time = None
paused = False


def gen_log_file_path(basename, ext="log"):
    """
    Generate a timestamped log file path.

    Args:
        basename (str): The base name of the log file (e.g., "log/performance/performance_log").
        ext (str): The file extension (default is "log").

    Returns:
        str: A file path including the timestamp, e.g., "performance_log_20250713_001530.log".
    """
    now = datetime.now()
    timestamp = now.strftime("%Y%m%d_%H%M%S")
    file_path = f"{basename}_{timestamp}.{ext}"
    return file_path


def init_log(file_path=None):
    """
    Initialize the performance log file.

    Args:
        file_path (str, optional): Path to the log file. If None, a default path is generated.
    """
    if file_path is None:
        file_path = gen_log_file_path(basename="log/performance/performance_log")
        os.makedirs(os.path.dirname(file_path), exist_ok=True)
    global log_file, last_time, paused
    log_file = open(file_path, "w")
    log_file.write("component, function, time [sec], per frame [sec], num frames\n")
    last_time = time.time()
    paused = False


def log_timing(component_name, func_name="", num_frames=None):
    """
    Log the time elapsed since the last log entry.

    Args:
        component_name (str): The name of the component (e.g., module or pipeline stage).
        func_name (str, optional): The name of the function (for fine-grained profiling).
        num_frames (int, optional): If provided, logs time per frame and frame count.
    """
    global last_time
    if log_file is None:
        return
    if paused:
        return  # Do nothing while logging is paused
    now = time.time()
    elapsed = now - last_time

    per_frame = ", "
    if num_frames is not None:
        per_frame = f"{elapsed / num_frames:.5f}, {num_frames}"
    log_file.write(f"{component_name}, {func_name}, {elapsed:.5f}, {per_frame}\n")
    log_file.flush()
    last_time = now


def pause_log():
    """
    Temporarily pause logging.

    While paused, `log_timing()` will not record any log entries.
    """
    global paused
    paused = True


def resume_log():
    """
    Resume logging after a pause.

    Resets the internal timer so the next call to `log_timing()` measures
    from this point onward.
    """
    global paused, last_time
    paused = False
    last_time = time.time()  # Reset timer on resume


def close_log():
    """
    Close the current log file safely.

    Ensures all contents are flushed and file handle is released.
    """
    global log_file
    if log_file is not None:
        log_file.close()
        log_file = None
