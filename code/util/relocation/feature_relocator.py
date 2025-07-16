# -----------------------------------------------------------------------------
# Stroke Transfer for Participating Media
# http://cg.it.aoyama.ac.jp/yonghao/sig25/abstsig25.html
#
# File: util/relocation/feature_relocator.py
# Maintainer: Hideki Todo
#
# Description:
# Feature-wise relocation module using octave noise with adaptive scaling based on frame distance.
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
import json
import os

import numpy as np

from util.common.feature_basis_def import FEATURE_NAMES
from util.infra.logger import log_debug
from util.relocation.relocation import logger, OctaveNoiseRelocator


class FeatureWiseOctaveRelocator:
    """
    Applies feature-wise octave noise relocation with adaptive scaling.

    Attributes:
        relocators (dict): Mapping from feature name to OctaveNoiseRelocator.
        r_scale_min (float): See args.
        r_scale_max (float): See args.
        sigma (float): See args.
        feature_names (List[str]): See args.
        learn_frames (List[int]): See args.
        scale_space (float): See args.
        scale_time (float): See args.
    """
    def __init__(self, r_scale_min, r_scale_max, sigma, feature_names, learn_frames,
                 scale_space, scale_time):
        """
        Initializes feature-dependent relocators with shared noise parameters.

        Args:
            r_scale_min (float): Minimum relocation scale.
            r_scale_max (float): Maximum relocation scale.
            sigma (float): Controls decay of r_scale based on frame distance.
            feature_names (list): List of feature names.
            learn_frames (list): Key frames used to adjust r_scale.
            scale_space (float): Spatial noise scale.
            scale_time (float): Temporal noise scale.
        """
        self.relocators = generate_feature_dependent_relocators(feature_names,
                                                                r_scale=0.05,
                                                                scale_space=scale_space,
                                                                scale_time=scale_time,
                                                                seed=1)
        self.r_scale_min = r_scale_min
        self.r_scale_max = r_scale_max
        self.sigma = sigma
        self.feature_names = feature_names
        self.learn_frames = learn_frames
        self.scale_space = scale_space
        self.scale_time = scale_time

    def relocate(self, feature_name, I, frame):
        """
        Applies feature-specific relocation based on frame distance.

        Args:
            feature_name (str): Name of the feature.
            I (ndarray): Input feature image.
            frame (int): Frame index.

        Returns:
            ndarray: Relocated feature.
        """
        # print(f"   - r_scale_min:  {self.r_scale_min}")
        # print(f"   - r_scale_max:  {self.r_scale_max}")
        # print(f"   - frame: {frame}")

        dist_frame = np.min(np.abs(np.array(self.learn_frames) - frame))
        r_scale_frame = (self.r_scale_max - self.r_scale_min) * np.exp(
            - dist_frame ** 2 / (2 * self.sigma ** 2)) + self.r_scale_min

        # print(f" - learn_frames: {self.learn_frames}")
        # print(f" - {frame}:  r_scale = {r_scale_frame}")
        #
        # print(f"   - dist_frame: {dist_frame}")

        relocator = self.relocators[feature_name]
        return relocator.relocate(I, frame, r_scale=r_scale_frame)

    def to_dict(self):
        """
        Serializes internal state to a dictionary.

        Returns:
            dict: Serialized parameters.
        """
        return {
            "r_scale_min": self.r_scale_min,
            "r_scale_max": self.r_scale_max,
            "sigma": self.sigma,
            "feature_names": self.feature_names,
            "learn_frames": self.learn_frames,
            "scale_space": self.scale_space,
            "scale_time": self.scale_time,
        }

    @classmethod
    def from_dict(cls, data):
        """
        Creates an instance from serialized parameters.

        Args:
            data (dict): Dictionary of parameters.

        Returns:
            FeatureWiseOctaveRelocator: Restored instance.
        """

        return cls(
            data["r_scale_min"],
            data["r_scale_max"],
            data["sigma"],
            data["feature_names"],
            data["learn_frames"],
            data["scale_space"],
            data["scale_time"],
        )


def generate_feature_dependent_relocators(
        feature_names,
        r_scale=0.05,
        scale_space=25,
        scale_time=50,
        seed=1):
    """
    Generates a relocator per feature using shared noise parameters.

    Args:
        feature_names (list): List of feature names.
        r_scale (float): Relocation scale.
        scale_space (float): Spatial scale of noise.
        scale_time (float): Temporal scale of noise.
        seed (int): Random seed.

    Returns:
        dict: Mapping of feature names to relocators.
    """
    relocators = {}

    np.random.seed(seed)

    for feature_name in feature_names:
        relocators[feature_name] = OctaveNoiseRelocator(
            r_scale=r_scale,
            scale_space=scale_space,
            scale_time=scale_time,
            shift_xyz=np.random.uniform(0, 200, size=(3, 3)))
    return relocators


def load_feature_dependent_relocator(in_file):
    """
    Loads a feature relocator from a JSON file.

    Args:
        in_file (str): Path to relocator JSON.

    Returns:
        OctaveNoiseFeaturesRelocator or None: Loaded instance or None if not found.
    """
    if not os.path.exists(in_file):
        return None

    with open(in_file, "r") as f:
        data = json.load(f)

    features_relocator = FeatureWiseOctaveRelocator.from_dict(data)
    return features_relocator


def save_feature_dependent_relocator(out_file, features_relocator):
    """
    Saves a feature relocator to a JSON file.

    Args:
        out_file (str): Output path.
        features_relocator (FeatureWiseOctaveRelocator): Relocator to save.
    """
    with open(out_file, "w") as f:
        json.dump(features_relocator.to_dict(), f, indent=4)


def prepare_feature_dependent_relocator(
        out_relocator_setting_file="relocate/octave_relocator.json",
        r_scale_min=0.01, r_scale_max=0.02, r_sigma=3,
        r_scale_space=100, r_scale_time=100000,
        learn_frames=[1],
        feature_names=FEATURE_NAMES):
    """
    Creates and saves a new OctaveNoiseFeaturesRelocator with default settings.

    Args:
        out_relocator_setting_file (str): Output JSON path.
        r_scale_min (float): Minimum relocation scale.
        r_scale_max (float): Maximum relocation scale.
        r_sigma (float): Frame distance decay factor.
        r_scale_space (float): Spatial noise scale.
        r_scale_time (float): Temporal noise scale.
        learn_frames (list): Reference frames.
        feature_names (list): List of feature names.
    """
    log_debug(logger, f"- learn_frames: {learn_frames}")
    log_debug(logger, f"- r_scale_min: {r_scale_min}")
    log_debug(logger, f"- r_scale_max: {r_scale_max}")
    log_debug(logger, f"- r_sigma: {r_sigma}")
    log_debug(logger, f"- r_scale_space: {r_scale_space}")
    log_debug(logger, f"- r_scale_time: {r_scale_time}")

    features_relocator = FeatureWiseOctaveRelocator(
        r_scale_min=r_scale_min, r_scale_max=r_scale_max,
        sigma=r_sigma, feature_names=feature_names,
        learn_frames=learn_frames,
        scale_space=r_scale_space, scale_time=r_scale_time,
    )

    out_dir = os.path.dirname(out_relocator_setting_file)
    os.makedirs(out_dir, exist_ok=True)
    save_feature_dependent_relocator(out_relocator_setting_file, features_relocator)
