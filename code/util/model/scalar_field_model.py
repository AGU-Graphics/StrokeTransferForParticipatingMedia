# -----------------------------------------------------------------------------
# Stroke Transfer for Participating Media
# http://cg.it.aoyama.ac.jp/yonghao/sig25/abstsig25.html
#
# File: util/model/scalar_field_model.py
# Maintainer: Hideki Todo
#
# Description:
# Regression models for transferring scalar or multi-channel fields (e.g., color, width, length)
# from feature sets using linear regression or k-NN (with optional Faiss acceleration).
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
import faiss
import numpy as np
from sklearn.linear_model import BayesianRidge
from sklearn.neighbors import KNeighborsRegressor
from sklearn.preprocessing import PolynomialFeatures
from sklearn.utils import shuffle

from util.infra.logger import log_subsection


class ScalarFieldRegressionModel:
    """Regression model for scalar fields (length, width).

    This model learns a mapping from proxy features to a scalar target field
    using polynomial feature expansion and Bayesian linear regression.

    Attributes:
        order (int): Polynomial order for feature expansion.
        feature_maps (dict): Dictionary of named proxy feature images.
        model (BayesianRidge): Regression model instance.
        polynomial_features (PolynomialFeatures): Feature expansion utility.
        target_image_shape (tuple): Shape of the input target image.
        I_min (float): Minimum value for clipping predictions.
        I_max (float): Maximum value for clipping predictions.
    """

    def __init__(self, order=1):
        """
        Args:
            order (int): Polynomial order for feature expansion.
        """
        log_subsection(f"Fit ScalarFieldRegressionModel(order={order})")
        self.order = order
        self.polynomial_features = PolynomialFeatures(order)

        self.feature_maps = {}

        self.model = BayesianRidge()
        self.target_image_shape = None
        self.I_min = None
        self.I_max = None

    def clean_internal(self):
        """Clear internal feature data for serialization (e.g., pickling)."""
        self.feature_maps = {}

    def set_feature(self, key, feature):
        """Registers a feature map with the given key.

        Args:
            key (str): Name of the feature.
            feature (np.ndarray): Feature data of shape (H, W).
        """
        self.feature_maps[key] = feature

    def set_mask(self, mask):
        """Sets the spatial mask indicating target regions.

        Args:
            mask (np.ndarray): Binary or float mask of shape (H, W).
        """
        self.mask = mask

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

    def fit(self, target_image, mask=None):
        """ Fit color/length/width field model for the target image.

        Args:
            target_image (np.ndarray): Target color/length/width image (H, W, D).
            mask (np.ndarray, optional): Optional binary mask of shape (H, W).
        """
        W_u_constraints, I_constraints = self.constraints(target_image, mask)
        self.fit_constraints(W_u_constraints, I_constraints)
        return

    def constraints(self, target_image, mask=None):
        """ Return the model constraints for multi-exemplars.

        Args:
            target_image (np.ndarray): Target color/length/width image (H, W, D).
            mask (np.ndarray, optional): Optional binary mask of shape (H, W).

        Returns:
            W_u_constraints: constraints for feature matrix.
            I_constraints: constraints for the target color/length/width.
        """
        self.target_image_shape = target_image.shape
        h, w = target_image.shape[:2]
        num_data = h * w

        W_u = self.get_feature_matrix()
        I_dash_flat = target_image.reshape(num_data, -1)

        if mask is not None:
            A_flat = mask.flatten()
            W_u = W_u[A_flat > 0.5, :]
            I_dash_flat = I_dash_flat[A_flat > 0.5, :]

        W_u_constraints = W_u
        I_constraints = I_dash_flat
        return W_u_constraints, I_constraints

    def fit_constraints(self, W_u_constraints, I_constraints):
        """ Fit length/width field model for the given constraints.

        Args:
            W_u_constraints: constraints for feature matrix.
            I_constraints: constraints for the target length/width.
        """
        if I_constraints.ndim == 2:
            if I_constraints.shape[1] == 1:
                I_constraints = I_constraints.ravel()
        self.model.fit(W_u_constraints, I_constraints)
        self.I_min = max(np.min(I_constraints), 0.001)
        self.I_max = np.max(I_constraints)
        return

    def predict(self, target_image_shape=None):
        """ predict length/width field using the model.

        Args:
            target_image_shape: target image size.
        Returns:
            I_predicted: Predicted length/width image (H, W, D).
        """
        if target_image_shape is None:
            target_image_shape = self.target_image_shape
        W_u = self.get_feature_matrix()

        I_predicted_flat = self.model.predict(W_u)

        if self.I_min is not None and self.I_max is not None:
            I_predicted_flat = np.clip(I_predicted_flat, self.I_min, self.I_max)

        I_predicted = I_predicted_flat.reshape(target_image_shape)

        return I_predicted


class ColorFieldRegressionModel:
    """Regression model for multi-channel color or attribute fields.

    Attributes:
        order (int): Polynomial order for feature expansion.
        models (list): List of ScalarFieldRegressionModel, one per channel.
        target_image_shape (tuple): Shape of the input target image.
    """

    def __init__(self, order=1):
        """
        Args:
            order (int): Polynomial order for feature expansion.
        """
        self.order = order
        self.models = []

        for ci in range(4):
            model = ScalarFieldRegressionModel(self.order)
            self.models.append(model)
        self.target_image_shape = None

    def set_feature(self, key, feature):
        """Registers a feature map with the given key.

        Args:
            key (str): Name of the feature.
            feature (np.ndarray): Feature data of shape (H, W).
        """
        for model in self.models:
            model.set_feature(key, feature)

    def fit(self, target_image, mask=None):
        """ Fit color field model for the target image.

        Args:
            target_image (np.ndarray): Target color image (H, W, D).
            mask (np.ndarray, optional): Optional binary mask of shape (H, W).
        """
        self.target_image_shape = target_image.shape
        h, w, cs = target_image.shape

        I_predicted = np.array(target_image)
        phi = []

        for ci in range(cs):
            model = self.models[ci]
            I_predicted_i, phi_i = model.fit(target_image[:, :, ci].reshape(h, w), mask)
            I_predicted[:, :, ci] = I_predicted_i
            phi.append(phi_i)

        return I_predicted, phi

    def predict(self, target_image_shape=None):
        """ predict color field using the model.

        Args:
            target_image_shape: target image size.
        Returns:
            I_predicted: Predicted color image (H, W, D).
        """
        if target_image_shape is None:
            target_image_shape = self.target_image_shape

        h, w, cs = target_image_shape

        I_predicted = np.zeros(target_image_shape)

        for ci in range(cs):
            I_predicted_i = self.models[ci].predict(target_image_shape[:2])
            I_predicted[:, :, ci] = I_predicted_i
        return I_predicted


class KNN_Regressor_faiss:
    """Approximate k-NN regressor using Faiss for fast neighbor search.

    Attributes:
        k (int): Number of neighbors to consider.
        faster_search (bool): Use IVF (coarse quantization) for speedup.
        gpu_mode (bool): Use GPU-accelerated Faiss if True.
    """
    def __init__(self, k=3, faster_search=True, gpu_mode=False):
        """
        Args:
            k (int): Number of neighbors to consider.
            faster_search (bool): Use IVF (coarse quantization) for speedup.
            gpu_mode (bool): Use GPU-accelerated Faiss if True.
        """
        self.k = k
        self.faster_search = faster_search
        self.gpu_mode = gpu_mode

    def fit(self, X, Y):
        """Train the Faiss index and store target values.

        Chooses between CPU/GPU and fast/standard indexing.
        """
        if self.gpu_mode:  # GPU mode
            if self.faster_search:
                self.fit_gpu_fast(X, Y)
            else:
                self.fit_gpu(X, Y)
        else:  # CPU mode
            if self.faster_search:
                self.fit_fast(X, Y)
            else:
                self.fit_base(X, Y)

    def fit_base(self, X, Y):
        """Build a brute-force index (L2) on CPU."""
        d = X.shape[1]  # dimension

        index = faiss.IndexFlatL2(d)  # build the index
        index.add(X)  # add vectors to the index

        self.index = index
        self.Y = Y

    def fit_fast(self, X, Y):
        """Build a fast IVF (Voronoi) index on CPU."""
        d = X.shape[1]  # dimension
        nlist = 10
        quantizer = faiss.IndexFlatL2(d)
        index = faiss.IndexIVFFlat(quantizer, d, nlist, faiss.METRIC_L2)  # fast: voronoi
        index.train(X)
        index.add(X)  # add vectors to the index
        self.index = index
        self.Y = Y

    def fit_gpu(self, X, Y):
        """Build a brute-force index on GPU."""
        res = faiss.StandardGpuResources()

        d = X.shape[1]  # dimension

        # build a flat (CPU) index
        index_flat = faiss.IndexFlatL2(d)
        # make it into a gpu index
        gpu_index_flat = faiss.index_cpu_to_gpu(res, 0, index_flat)
        gpu_index_flat.add(X)
        # self.gpu_index_flat = gpu_index_flat
        self.index = faiss.index_gpu_to_cpu(gpu_index_flat)
        self.Y = Y

    def fit_gpu_fast(self, X, Y):
        """Build a fast IVF (Voronoi) index on GPU."""
        res = faiss.StandardGpuResources()
        d = X.shape[1]
        nlist = 10
        quantizer = faiss.IndexFlatL2(d)
        index = faiss.IndexIVFFlat(quantizer, d, nlist, faiss.METRIC_L2)
        gpu_index_flat = faiss.index_cpu_to_gpu(res, 0, index)
        gpu_index_flat.train(X)
        gpu_index_flat.add(X)

        # self.gpu_index_flat = gpu_index_flat
        self.index = faiss.index_gpu_to_cpu(gpu_index_flat)
        self.Y = Y

    def predict(self, X_query):
        """Perform k-NN prediction on the given query points."""
        if self.gpu_mode:
            return self.predict_gpu(X_query)
        else:
            return self.predict_cpu(X_query)

    def _gather_predictions(self, indices):
        """Aggregate neighbor targets using mean (or single match)."""
        if self.k == 1:
            return self.Y[indices]
        else:
            return np.mean(self.Y[indices], axis=1)

    def predict_cpu(self, X_query):
        """Predict using the CPU-based Faiss index."""
        D, I = self.index.search(X_query, self.k)

        return self._gather_predictions(I)

    def predict_gpu(self, X_query):
        """Predict using the GPU-based Faiss index."""
        res = faiss.StandardGpuResources()
        gpu_index_flat = faiss.index_cpu_to_gpu(res, 0, self.index)

        D, I = gpu_index_flat.search(X_query, self.k)

        return self._gather_predictions(I)


class NearestNeighborModel:
    """Nearest neighbor regression model for color/length/width fields.

    Attributes:
        feature_maps (dict): Dictionary of named proxy feature images.
        model: Backend k-NN model (KNeighborsRegressor or KNN_Regressor_faiss).
        num_samples (int): Number of sample points used during fitting.
    """

    def __init__(self, num_samples=2000, use_faiss=False, gpu_mode=False):
        """
        Args:
            num_samples (int): number of sampling constraints.
            use_faiss (bool): turn on faiss mode.
            gpu_mode (bool): turn on gpu mode.
        """
        log_subsection(
            f"Fit NearestNeighborModel(num_samples={num_samples}, use_faiss={use_faiss}, gpu_mode={gpu_mode})")
        self.feature_maps = {}

        if use_faiss:
            self.model = KNN_Regressor_faiss(k=3, faster_search=True, gpu_mode=gpu_mode)
        else:
            self.model = KNeighborsRegressor(n_neighbors=3, algorithm="kd_tree",
                                             leaf_size=30, n_jobs=10)

        self.num_samples = num_samples

    def set_feature(self, key, feature):
        """Registers a feature map with the given key.

        Args:
            key (str): Name of the feature.
            feature (np.ndarray): Feature data of shape (H, W).
        """
        self.feature_maps[key] = feature

    def clean_internal(self):
        """Clear internal feature data for serialization (e.g., pickling)."""
        self.feature_maps = {}

    def get_feature_matrix(self):
        """Constructs polynomial-expanded feature matrix W_u.

        Returns:
            np.ndarray: Expanded feature matrix of shape (H*W, N_features_poly).
        """
        W_u = []

        for key, feature_k in self.feature_maps.items():
            W_u.append(feature_k.flatten())

        W_u = np.array(W_u).T
        return W_u

    def fit(self, target_image, mask=None):
        """ Fit color/length/width field model for the target image.

        Args:
            target_image (np.ndarray): Target color/length/width image (H, W, D).
            mask (np.ndarray, optional): Optional binary mask of shape (H, W).
        """
        W_u_constraints, I_constraints = self.constraints(target_image, mask)
        self.fit_constraints(W_u_constraints, I_constraints)
        return

    def constraints(self, target_image, mask=None):
        """ Return the model constraints for multi-exemplars.

        Args:
            target_image (np.ndarray): Target color/length/width image (H, W, D).
            mask (np.ndarray, optional): Optional binary mask of shape (H, W).

        Returns:
            W_u_constraints: constraints for feature matrix.
            I_constraints: constraints for the target color/length/width.
        """
        self.target_image_shape = target_image.shape
        h, w = target_image.shape[:2]
        num_data = h * w

        W_u = self.get_feature_matrix()
        I_dash_flat = target_image.reshape(num_data, -1)

        if mask is not None:
            A_flat = mask.flatten()
            W_u = W_u[A_flat > 0.5, :]
            I_dash_flat = I_dash_flat[A_flat > 0.5, :]

        num_samples = self.num_samples
        if W_u.shape[0] > num_samples:
            W_u_constraints = shuffle(W_u, random_state=0, n_samples=num_samples)
            I_constraints = shuffle(I_dash_flat, random_state=0, n_samples=num_samples)
        else:
            W_u_constraints = W_u
            I_constraints = I_dash_flat
        return W_u_constraints, I_constraints

    def fit_constraints(self, W_u_constraints, I_constraints):
        """ Fit color/length/width field model for the given constraints.

        Args:
            W_u_constraints: constraints for feature matrix.
            I_constraints: constraints for the target color/length/width.
        """
        self.model.fit(W_u_constraints, I_constraints)
        return

    def predict(self, target_image_shape=None):
        """ predict color/length/width field using the model.

        Args:
            target_image_shape: target image size.
        Returns:
            I_predicted: Predicted color/length/width image (H, W, D).
        """
        if target_image_shape is None:
            target_image_shape = self.target_image_shape

        W_u = self.get_feature_matrix()

        I_predicted_flat = self.model.predict(W_u)
        I_predicted = I_predicted_flat.reshape(target_image_shape)

        return I_predicted
