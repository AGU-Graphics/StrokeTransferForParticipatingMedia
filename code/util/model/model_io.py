# -----------------------------------------------------------------------------
# Stroke Transfer for Participating Media
# http://cg.it.aoyama.ac.jp/yonghao/sig25/abstsig25.html
#
# File: util/model/model_io.py
# Maintainer: Hideki Todo
#
# Description:
# Utility functions to save and load regression models (e.g., orientation, color, length, width) using Python's pickle format.
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
import pickle

from util.infra.logger import getLogger

logger = getLogger()


def load_model(model_file):
    """Load a regression model (e.g., orientation, color, length, or width) from a pickle file.

    Args:
        model_file (str): Path to the pickle file.

    Returns:
        model: Deserialized regression model object.
    """
    with open(model_file, 'rb') as f:
        model = pickle.load(f)
    return model


def save_model(model, model_file):
    """Save a regression model (e.g., orientation, color, length, or width) to a pickle file.

    Args:
        model (Any): Trained regression model to serialize.
        model_file (str): Destination path to save the pickle file.
    """
    model_dir = os.path.dirname(model_file)
    os.makedirs(model_dir, exist_ok=True)
    with open(model_file, 'wb') as f:
        pickle.dump(model, f)
