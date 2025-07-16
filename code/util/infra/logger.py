# -----------------------------------------------------------------------------
# Stroke Transfer for Participating Media
# http://cg.it.aoyama.ac.jp/yonghao/sig25/abstsig25.html
#
# File: util/infra/logger.py
# Maintainer: Hideki Todo
#
# Description:
# Logging utilities for consistent and structured output in pipeline stages.
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
import logging
import sys

# Default logging level
_current_level = logging.INFO


class ConditionalFormatter(logging.Formatter):
    """
    Custom formatter that includes logger name only in DEBUG level.
    """

    def format(self, record):
        # self._style._fmt = "[%(levelname)s] %(message)s"
        self._style._fmt = "%(message)s"
        return super().format(record)


# Dictionary to track all created loggers
_registered_loggers = {}


def getLogger(name="stroke_transfer"):
    """
    Get a configured logger with consistent formatting.

    Prevents adding duplicate handlers for the same logger name.

    Args:
        name (str): The name of the logger.

    Returns:
        logging.Logger: Configured logger instance.
    """
    logger = logging.getLogger(name)
    logger.setLevel(_current_level)

    if name not in _registered_loggers:
        handler = logging.StreamHandler(sys.stdout)
        handler.setFormatter(ConditionalFormatter())
        logger.addHandler(handler)
        _registered_loggers[name] = logger

    return logger


def set_level(level):
    """
    Set the logging level globally for all registered loggers.

    Args:
        level (int): Logging level (e.g., logging.DEBUG, logging.INFO).
    """
    global _current_level
    _current_level = level
    for logger in _registered_loggers.values():
        logger.setLevel(level)


def log_section(title):
    """
    Print a major visual section header.

    Args:
        title (str): The title of the section.
    """
    line = "-" * 60
    print("")  # Insert blank line for spacing
    print(line)
    print(title)
    print(line)


def log_section_start(title):
    """
    Print a banner indicating the start of a section.

    Args:
        title (str): The section title.
    """
    print("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━")
    print(f"[Start] {title}")


def log_section_done(title, duration):
    """
    Print a banner indicating the end of a section with timing info.

    Args:
        title (str): The section title.
        duration (float): Time taken for the section in seconds.
    """
    print(f"\n[Done] {title} in {duration:.2f} sec")
    print("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━")


def log_phase(title):
    """
    Print a labeled pipeline phase header.

    Args:
        title (str): Phase name.
    """
    print(f"\n== {title} ==")


def log_subsection(subtitle):
    """
    Print a minor visual subsection label.

    Args:
        subtitle (str): Subsection title.
    """
    print(f"  -- {subtitle} --")


def _log_with_indent(logger_method, message, level_label, indent=2):
    """
    Internal utility for indented logging.

    Args:
        logger_method (Callable): Method to call (e.g., logger.info).
        message (str): The message to log.
        level_label (str): Label to prefix (e.g., DEBUG, INFO).
        indent (int): Indentation level.
    """
    prefix = "  " * indent
    logger_method(f"{prefix}[{level_label}] {message}")


def log_debug(logger, message, indent=2):
    """
    Log a debug-level message with indentation.

    Args:
        logger (logging.Logger): Logger to use.
        message (str): Message to log.
        indent (int): Indentation level.
    """
    _log_with_indent(logger.debug, message, "DEBUG", indent)


def log_info(logger, message, indent=2):
    """
    Log an info-level message with indentation.

    Args:
        logger (logging.Logger): Logger to use.
        message (str): Message to log.
        indent (int): Indentation level.
    """
    _log_with_indent(logger.info, message, "INFO", indent)


def log_warning(logger, message, indent=2):
    """
    Log a warning-level message with indentation.

    Args:
        logger (logging.Logger): Logger to use.
        message (str): Message to log.
        indent (int): Indentation level.
    """
    _log_with_indent(logger.warning, message, "WARNING", indent)


def log_out_files(logger, out_files, indent=2, level="debug"):
    """
    Log a list of output file paths.

    Args:
        logger (logging.Logger): Logger to use.
        out_files (List[str]): List of output file paths.
        indent (int): Indentation level.
        level (str): Logging level ("debug" or "info").
    """
    message = "Output Files:"
    prefix = "  " * (indent + 1)
    for out_file in out_files:
        message += f"\n{prefix}- {out_file}"

    if level == "debug":
        log_debug(logger, message, indent=indent)
    elif level == "info":
        log_info(logger, message, indent=indent)


def log_in_files(logger, in_files, indent=2, level="debug"):
    """
    Log a list of input file paths.

    Args:
        logger (logging.Logger): Logger to use.
        in_files (List[str]): List of input file paths.
        indent (int): Indentation level.
        level (str): Logging level ("debug" or "info").
    """
    message = "Input Files:"
    prefix = "  " * (indent + 1)
    for in_file in in_files:
        message += f"\n{prefix}- {in_file}"

    if level == "debug":
        log_debug(logger, message, indent=indent)
    elif level == "info":
        log_info(logger, message, indent=indent)
