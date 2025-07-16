# -----------------------------------------------------------------------------
# Stroke Transfer for Participating Media
# http://cg.it.aoyama.ac.jp/yonghao/sig25/abstsig25.html
#
# File: util/annotation_interpolation/rbf_model.py
# Maintainer: Hideki Todo
#
# Description:
# Implements a multi-output Radial Basis Function (RBF) regression model.
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
from scipy.interpolate import Rbf
from sklearn.utils import shuffle


class RBFModel:
    def __init__(self, max_samples=1000000, smoothness=1e-3):
        """
        Radial Basis Function (RBF) regression model with optional subsampling.

        Args:
            max_samples (int): Maximum number of input samples to use during training.
            smoothness (float): Smoothing parameter for RBF interpolation.
        """
        self.max_samples = max_samples
        self.smoothness = smoothness

    def fit(self, X, Y):
        """
        Fit RBF models to each output dimension in Y based on inputs X.

        Args:
            X (np.ndarray): Input feature array of shape (N, D).
            Y (np.ndarray): Output target array of shape (N, C).
        """
        XY = np.hstack([X, Y])
        if XY.shape[0] > self.max_samples:
            samples = shuffle(XY, random_state=0)[:self.max_samples]
            X_samples = samples[:, :X.shape[1]]
            Y_samples = samples[:, X.shape[1]:]
        else:
            X_samples = np.array(X)
            Y_samples = np.array(Y)

        self.rbfs = []
        for dim in range(Y_samples.shape[1]):
            input_dims = [X_samples[:, d] for d in range(X_samples.shape[1])]
            output_dim = Y_samples[:, dim]
            rbf_func = Rbf(*input_dims, output_dim, smooth=self.smoothness)
            self.rbfs.append(rbf_func)

    def transform(self, X):
        """
        Predict output values for new input data using the trained RBF models.

        Args:
            X (np.ndarray): New input feature array of shape (N, D).

        Returns:
            Y (np.ndarray): Predicted output array of shape (N, C).
        """
        input_dims = [X[:, d] for d in range(X.shape[1])]
        Y = np.zeros((X.shape[0], len(self.rbfs)))

        for dim, rbf_func in enumerate(self.rbfs):
            Y[:, dim] = rbf_func(*input_dims)
        return Y
