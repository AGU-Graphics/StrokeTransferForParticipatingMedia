# -----------------------------------------------------------------------------
# Stroke Transfer for Participating Media
# http://cg.it.aoyama.ac.jp/yonghao/sig25/abstsig25.html
#
# File: util/infra/timer.py
# Maintainer: Hideki Todo
#
# Description:
# Utility class for timing and logging execution performance.
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
import time

from util.infra.logger import getLogger, log_debug, log_info

logger = getLogger()


class Timer:
    """A context-aware timer for measuring execution time with formatted logging.

    Attributes:
        name (str): Label for the timed section.
        level (int): Indentation level for nested timing.
        log_level (str): Logging level to use ("debug" or "info").
    """

    def __init__(self, name="Timer", level=0, log_level="debug"):
        """Initialize the timer.

        Args:
            name (str): Name of the timer to include in logs.
            level (int): Indentation level for log readability.
            log_level (str): Logging method to use, "debug" or "info".
        """
        self.name = name
        self.level = level
        self.log_level = log_level
        self.start_time = None
        self.end_time = None

    def _indent(self, msg):
        """Apply indentation based on `self.level`.

        Args:
            msg (str): Message to indent.

        Returns:
            str: Indented message.
        """
        return "  " * self.level + msg

    def _log(self, msg):
        """Log a message using the configured logging level.

        Args:
            msg (str): Message to log.
        """
        if self.log_level == "info":
            log_info(logger, self._indent(msg))
        else:
            log_debug(logger, self._indent(msg))

    def __enter__(self):
        """Start timing when entering a `with` block.

        Returns:
            Timer: The timer instance.
        """
        self.start()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Stop timing when exiting a `with` block."""
        self.stop()

    def start(self):
        """Start the timer and log the start event."""
        self.start_time = time.time()
        self._log(f"== {self.name}: (Start) ==")

    def stop(self):
        """Stop the timer and log the elapsed time."""
        self.end_time = time.time()
        elapsed = self.end_time - self.start_time
        self._log(f"== {self.name}: (Done) {elapsed:.4f} sec ==")

    def elapsed(self):
        """Get the current elapsed time since start.

        Returns:
            float: Time in seconds since the timer was started. If not started, returns 0.0.
        """
        return time.time() - self.start_time if self.start_time else 0.0
