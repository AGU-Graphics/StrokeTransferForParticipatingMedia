# -----------------------------------------------------------------------------
# Stroke Transfer for Participating Media
# http://cg.it.aoyama.ac.jp/yonghao/sig25/abstsig25.html
#
# File: util/model/orientation_model.py
# Maintainer: Hideki Todo
#
# Description:
# Provides a regression framework for transferring dense vector fields.
# from basis sets and proxy feature sets using linear regression.
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
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures

from util.infra.logger import getLogger, log_subsection
from util.normalize.norm import normalize_vector_image

logger = getLogger(__name__)


class VectorFieldRegressionModel:
    """Regression model to predict vector fields from basis and feature maps.

    Attributes:
        order (int): Polynomial order for the feature expansion.
        model: scikit-learn linear regression model.
        basis_maps (dict): Dictionary mapping basis names to (H, W, D) arrays.
        feature_maps (dict): Dictionary mapping feature names to (H, W) arrays.
        polynomial_features: PolynomialFeatures transformer for feature expansion.
    """

    def __init__(self, order=1):
        """
        Args:
            order (int): order of the vector field regression model.
        """
        log_subsection(f"Fit VectorFieldRegressionModel(order={order})")
        self.order = order
        # self.model = BayesianRidge(fit_intercept=False)
        self.model = LinearRegression(fit_intercept=False)

        self.basis_maps = {}
        self.basis_shape = None
        self.feature_maps = {}

        self.polynomial_features = PolynomialFeatures(order)

    def set_basis(self, key, basis):
        """Registers a basis map with the given key.

        Args:
            key (str): Name of the basis component.
            basis (np.ndarray): Basis data of shape (H, W, D).
        """
        self.basis_maps[key] = basis
        self.basis_shape = basis.shape

    def set_feature(self, key, feature):
        """Registers a feature map with the given key.

        Args:
            key (str): Name of the feature.
            feature (np.ndarray): Feature data of shape (H, W).
        """
        self.feature_maps[key] = feature

    def clean_internal(self):
        """Clears internal basis and feature maps (for saving or reuse)."""
        self.basis_maps = {}
        self.feature_maps = {}

    def set_mask(self, mask):
        """Sets the spatial mask indicating target regions.

        Args:
            mask (np.ndarray): Binary or float mask of shape (H, W).
        """
        self.mask = mask

    def get_basis_matrix(self):
        """Constructs basis matrix A_hat.

        Returns:
            np.ndarray: Flattened basis matrix of shape (D*H*W, N_basis).
        """
        A_hat = []

        for key, A_i in self.basis_maps.items():
            A_hat.append(A_i[:, :, :3].flatten())

        A_hat = np.array(A_hat)
        A_hat = A_hat.T

        return A_hat

    def get_feature_matrix(self):
        """Constructs polynomial-expanded feature matrix W_u.

        Returns:
            np.ndarray: Expanded feature matrix of shape (H*W, N_features_poly).
        """
        W_u = []

        for key, feature_k in self.feature_maps.items():
            W_u.append(feature_k.flatten())

        W_u = np.array(W_u).T
        W_u = self.polynomial_features.fit_transform(W_u)

        return W_u

    def get_regression_matrix(self):
        """Constructs combined basis-feature regression matrix.

        Returns:
            np.ndarray: Regression matrix of shape (D*H*W, N_basis * N_features_poly).
        """
        A_hat = self.get_basis_matrix()

        if self.order == 0:
            return A_hat

        W = self.get_feature_matrix()

        AW = []

        for j in range(A_hat.shape[1]):
            for k in range(W.shape[1]):
                W_k = np.repeat(W[:, k], 3).flatten()
                AW.append(W_k * A_hat[:, j])

        AW = np.array(AW).T
        return AW

    def constraints(self, target_orientation, mask=None):
        """ Return the model constraints for multi-exemplars.

        Args:
            target_orientation (np.ndarray): Target vector field (H, W, D).
            mask (np.ndarray, optional): Optional mask of shape (H, W).

        Returns:
            AW_constraints: constraints for A_hat W_u matrix.
            u_constraints: constraints for the target orientation.
        """
        AW_constraints = self.get_regression_matrix()

        if mask is not None:
            mask_flat = np.repeat(mask.flatten(), 3).flatten()
            u_dash_flat = target_orientation[:, :, :3].flatten()
            return AW_constraints[mask_flat > 0.5, :], u_dash_flat[mask_flat > 0.5]
        else:
            return AW_constraints, target_orientation[:, :, :3].flatten()

    def fit_constraints(self, AW_constraints, u_constraints):
        """ Fits the regression model using precomputed constraints.

        Args:
            AW_constraints (np.ndarray): Regression matrix.
            u_constraints (np.ndarray): Flattened orientation targets.
        """
        self.model.fit(AW_constraints, u_constraints)

        self.phi = self.model.coef_
        return

    def fit(self, target_orientation, mask=None):
        """ Fits the regression model to the provided orientation and mask.

        Args:
            target_orientation (np.ndarray): Ground truth vector field (H, W, D).
            mask (np.ndarray, optional): Optional binary mask of shape (H, W).
        Returns:
            u_fit: Predicted orientation
            phi: Regression model weights.
        """
        AW_constraints, u_constraints = self.constraints(target_orientation, mask)
        self.fit_constraints(AW_constraints, u_constraints)
        u_predicted = self.predict()

        return u_predicted, self.phi

    def predict(self):
        """ Predicts a vector field using the fitted regression model.

        Returns:
            u_predicted: Predicted orientation field of shape (H, W, D).
        """
        AW = self.get_regression_matrix()

        u_flat = self.model.predict(AW)

        u_predicted = np.zeros(self.basis_shape)

        u_predicted[:, :, :3] = u_flat.reshape(self.basis_shape[0], self.basis_shape[1], 3)
        u_predicted[:, :, :2] = normalize_vector_image(u_predicted[:, :, :2])

        return u_predicted

    def compute_local_weight_map(self):
        """ Computes local weight contribution for each basis.

        Returns:
            weight_map: Weight maps for each basis of shape (H, W).
        """
        phi = self.phi
        mask = self.mask

        phi = phi.reshape(len(self.basis_maps.items()), -1)
        weight_map = {}

        basis_maps = self.basis_maps
        feature_maps = self.feature_maps

        for i, c_key in enumerate(basis_maps.keys()):
            w_sum = np.zeros_like(mask)
            for j, f_key in enumerate(feature_maps.keys()):
                Xj = feature_maps[f_key]
                Wi = phi[i, j + 1] * Xj
                w_sum += Wi

            w_sum += phi[i, 0]
            weight_map[c_key] = w_sum
        return weight_map

    def model_matrix(self, feature_names, basis_names):
        """ Return the model matrix for 1st-order model.

        Args:
            feature_names (list[str]): Names of registered features.
            basis_names (list[str]): Names of registered bases.
        Returns:
            phi_matrix: Matrix of shape (N_basis, N_features + 1).

        Note:
            Only support 1st-order mode.
        """
        phi = self.model.coef_
        phi = phi.reshape(len(self.basis_maps.items()), -1)
        self.phi = phi

        basis_keys = self.basis_maps.keys()
        feature_keys = self.feature_maps.keys()

        phi_matrix = np.zeros((len(basis_names), len(feature_names) + 1))

        for i, basis in enumerate(basis_keys):
            k = basis_names.index(basis)
            phi_matrix[k, -1] = phi[i, 0]
            for j, feature in enumerate(feature_keys):
                l = feature_names.index(feature)
                phi_matrix[k, l] = phi[i, j + 1]

        return phi_matrix


def scale_vector_field_by_mask(vector_field, mask):
    """Scales each vector in a vector field by the given spatial mask.

    Args:
        vector_field (np.ndarray): Input vector field of shape (H, W, D).
        mask (np.ndarray): Scaling mask of shape (H, W).

    Returns:
        np.ndarray: Scaled vector field.
    """
    V_ = np.array(vector_field)

    for ci in range(3):
        V_[:, :, ci] *= mask
    return V_


def project_vector_field(vector_field, normal):
    """Projects a vector field onto the plane orthogonal to the given normals.

    Args:
        vector_field (np.ndarray): Vector field (H, W, D).
        normal (np.ndarray): Normal vectors (H, W, D), usually D=3.

    Returns:
        np.ndarray: Projected vector field of shape (H, W, D).
    """
    udN = np.einsum("ijk,ijk->ij", vector_field[:, :, :3], normal[:, :, :3])
    u_proj = vector_field - np.einsum("ij,ijk->ijk", udN, normal)
    u_proj[:, :, 3] = vector_field[:, :, 3]
    return u_proj
