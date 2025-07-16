# -----------------------------------------------------------------------------
# Stroke Transfer for Participating Media
# http://cg.it.aoyama.ac.jp/yonghao/sig25/abstsig25.html
#
# File: util/exemplar_estimation/exemplar_estimation_GMM.py
# Maintainer: Yonghao Yue and Hideki Todo
#
# Description:
# Greedy exemplar frame estimation based on GMM fitting and BIC optimization for feature space sampling.
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
import itertools
import json
import os
import threading

import cv2
import numpy as np
import psutil
import tqdm
from sklearn.mixture import GaussianMixture

from util.common.feature_basis_io import load_hdf5
from util.exemplar_estimation.plot import plot_BIC_scores, plot_BIC_scores_for_optimal_exemplars, \
    plot_GMM_density_set
from util.infra.logger import log_subsection, getLogger, log_debug, log_info
from util.infra.timer import Timer

logger = getLogger()


def lmbd_from_a(a=1 / 6, num_fit_GMM=15, num_parameters=16):
    """Compute regularization constant lambda from scaling factor 'a'.

    Args:
        a (float): Scaling factor.
        num_fit_GMM (int): Number of GMM components.
        num_parameters (int): Feature dimension.

    Returns:
        float: regularization constant lambda.
    """
    lmbd = a * (num_fit_GMM - 1) * num_parameters * (num_parameters + 3) / 2
    return lmbd


def a_from_lmbd(lmbd=100, num_fit_GMM=15, num_parameters=16):
    """Compute scaling factor 'a' from regularization constant lambda.

    Args:
        lmbd (float): regularization constant lambda.
        num_fit_GMM (int): Number of GMM components.
        num_parameters (int): Feature dimension.

    Returns:
        float: Scaling factor 'a'.
    """
    a = lmbd / ((num_fit_GMM - 1) * num_parameters * (num_parameters + 3) / 2)
    return a


def save_json(out_file, data):
    """Save data to a JSON file.

    Args:
        out_file (str): Output file path.
        data (dict): Dictionary to save.
    """
    with open(out_file, 'w') as json_file:
        json.dump(data, json_file, indent=4)


def load_feature_image(feature_file_template, frame, resolution=None):
    """Load a single feature image for a given frame.

    Args:
        feature_file_template (str): Template for feature file path.
        frame (int): Frame index.
        resolution (tuple[int, int], optional): Target resolution (width, height).

    Returns:
        np.ndarray: Loaded (and optionally resized) feature image.
    """
    if feature_file_template is None:
        return None
    feature_file = feature_file_template % frame
    if not os.path.exists(feature_file):
        raise FileNotFoundError(f"File not found: {feature_file}")

    feature = load_hdf5(feature_file)
    if resolution is not None:
        feature = cv2.resize(feature, resolution)
    return feature


def load_feature_images(feature_file_templates, frame, resolution=None):
    """Load and stack multiple feature images for a given frame.

    Args:
        feature_file_templates (dict): Dictionary of feature name to file template.
        frame (int): Frame index.
        resolution (tuple[int, int], optional): Target resolution (width, height).

    Returns:
        np.ndarray or None: Stacked feature image or None if all are missing.
    """
    feature_images = []

    all_none = True
    for feature_name in feature_file_templates.keys():
        feature_file_template = feature_file_templates[feature_name]
        feature = load_feature_image(feature_file_template, frame, resolution=resolution)

        if feature is None:
            continue

        all_none = False

        feature_images.append(feature)

    if all_none:
        return None

    if len(feature_images) > 0:
        feature_images = np.dstack(feature_images)
    else:
        feature_images = feature_images[0]
        h, w = feature_images.shape[:2]
        feature_images = feature_images.reshape(h, w, 1)
    return feature_images


def feature_range_check(all_samples, feature_names):
    """Print statistical range information for each feature.

    Args:
        all_samples (np.ndarray): Sampled feature vectors.
        feature_names (list[str]): Names of each feature dimension.
    """
    for i, feature_name in enumerate(feature_names):
        feature_samples = all_samples[:, i]
        print(f"- {feature_names[i]}")
        print(f"  - (min, max): {np.min(feature_samples), np.max(feature_samples)}")
        print(f"  - (std): {np.std(feature_samples)}")


def sample_features_from_frame(
        feature_file_templates,
        frame=1,
        num_per_samples_frames=None,
        resolution=None):
    """Sample feature vectors from a single frame.

    Args:
        feature_file_templates (dict): Mapping of feature name to file template.
        frame (int): Frame index to sample.
        num_per_samples_frames (int, optional): Number of samples to draw randomly.
        resolution (tuple[int, int], optional): Resize resolution.

    Returns:
        np.ndarray: Sampled feature vectors.
    """
    feature = load_feature_images(feature_file_templates, frame, resolution)
    h, w = feature.shape[:2]
    num_pixels = h * w
    feature_sample = feature.reshape(num_pixels, -1)

    if num_per_samples_frames is not None:
        sample_ids = np.random.randint(low=0, high=num_pixels - 1, size=num_per_samples_frames)
        feature_sample = feature_sample[sample_ids, :]

    return feature_sample


def sample_features_from_frames(
        feature_file_templates,
        frames=[1],
        num_per_samples_frames=None,
        resolution=None):
    """Sample feature vectors from multiple frames.

    Args:
        feature_file_templates (dict): Mapping of feature name to file template.
        frames (list[int]): List of frame indices to sample from.
        num_per_samples_frames (int, optional): Number of samples per frame.
        resolution (tuple[int, int], optional): Resize resolution.

    Returns:
        np.ndarray: Feature samples of shape (num_frames, num_samples, num_features).
    """
    feature_samples = []

    for frame in tqdm.tqdm(frames, desc="Sampling"):
        feature_sample = sample_features_from_frame(
            feature_file_templates,
            frame=frame,
            num_per_samples_frames=num_per_samples_frames,
            resolution=resolution)

        feature_samples.append(feature_sample)

    feature_samples = np.array(feature_samples)

    return feature_samples


def fit_GMM_per_frame(feature_samples, num_fit_GMM=15):
    """Fit a Gaussian Mixture Model (GMM) to feature samples for each frame.

    Args:
        feature_samples (np.ndarray): Array of feature samples per frame.
        num_fit_GMM (int): Number of GMM components.

    Returns:
        list[GaussianMixture]: List of fitted GMM models.
    """

    log_subsection("Fit GMM for feature samples")

    num_frames = feature_samples.shape[0]
    fitted_GMM = []
    for f in tqdm.tqdm(range(num_frames), desc="Fit GMM (per frame)"):
        gm = GaussianMixture(n_components=num_fit_GMM, random_state=0).fit(feature_samples[f])
        fitted_GMM.append(gm)
    return fitted_GMM


def fit_GMM_runner(_points, _num_fit_GMM, parallel=True):
    """Fit GMMs to multiple sets of feature samples, optionally in parallel.

    Args:
        _points (list[np.ndarray]): List of feature samples.
        _num_fit_GMM (int): Number of GMM components.
        parallel (bool): Whether to use multi-threading.

    Returns:
        list[GaussianMixture]: Fitted GMM models.
    """
    _fit_data = []
    for f in range(len(_points)):
        _fit_data.append(GaussianMixture(n_components=_num_fit_GMM, random_state=0))

    if parallel:
        class FitThread(threading.Thread):
            def __init__(self, gm, points):
                super(FitThread, self).__init__()
                self.gm = gm
                self.points = points

            def run(self):
                self.gm.fit(self.points)

        num_threads = psutil.cpu_count()
        log_debug(logger, f"num_threads: {num_threads}")

        num_frames = len(_points)
        c = 0
        while True:
            if c >= num_frames:
                break

            _num_threads = min([num_threads, num_frames - c])

            threads = []
            for i in range(_num_threads):
                t = FitThread(_fit_data[c + i], _points[c + i])
                t.start()
                threads.append(t)
            for i in range(_num_threads):
                threads[i].join()

            c += _num_threads

    else:
        for f in tqdm.tqdm(range(len(_points))):
            _fit_data[f].fit(_points[f])

    return _fit_data


def convert_GMMmodels_to_GMMdata(_fit_data, _num_fit_GMM):
    """Convert trained GMM models to structured GMM parameter data.

    Args:
        _fit_data (list[GaussianMixture]): List of fitted GMM models.
        _num_fit_GMM (int): Number of components per GMM.

    Returns:
        list[list]: Per-frame list of [weight, mean, covariance] for each GMM component.
    """
    _GMM_per_frame = []

    for f in range(len(_fit_data)):
        _per_frame = []
        for g in range(_num_fit_GMM):
            _per_frame.append([_fit_data[f].weights_[g], _fit_data[f].means_[g], _fit_data[f].covariances_[g]])
        _GMM_per_frame.append(_per_frame)

    return _GMM_per_frame


def prepare_all_samples(_points):
    """Concatenate feature samples from all frames into a single array.

    Args:
        _points (list[np.ndarray]): List of feature arrays from multiple frames.

    Returns:
        np.ndarray: Concatenated feature samples.
    """
    _all_points = _points[0]
    for i in range(len(_points) - 1):
        _all_points = np.concatenate([_all_points, _points[i + 1]], axis=0)

    return _all_points


def GMM_all(GMM_per_frames):
    """Flatten list of per-frame GMM data.

    Args:
        GMM_per_frames (list[list]): List of GMM parameter sets per frame.

    Returns:
        list[list]: Flattened copy of GMM_per_frames.
    """
    GMM_frames = []
    for i in range(len(GMM_per_frames)):
        GMM_frames.append(GMM_per_frames[i])
    return GMM_frames


def search_optimal_num_fit_GMM(points, lmbd, parallel=True):
    """Search for the optimal number of GMM components based on BIC.

    Args:
        points (list[np.ndarray]): Feature samples per frame.
        lmbd (float): Regularization parameter.
        parallel (bool): Whether to use parallel processing.

    Returns:
        int: Optimal number of GMM components.
    """
    log_subsection("Search Optimal Number of GMM Components")
    all_points = prepare_all_samples(points)
    # fit_scores = []
    BIC_scores = []
    min_BIC_score = 1.0e33
    max_num_fit_GMM = 64
    best_num_fit_GMM = 1
    for i in range(max_num_fit_GMM):
        fitted_GMM = fit_GMM_runner(points, i + 1, parallel=parallel)
        GMM_per_frames = convert_GMMmodels_to_GMMdata(fitted_GMM, i + 1)
        GMM_all_frames = GMM_all(GMM_per_frames)

        if parallel:
            BIC_score = BICn_parallel(GMM_all_frames, all_points, lmbd)
        else:
            BIC_score = BICn(GMM_all_frames, all_points, lmbd)
        BIC_scores.append(BIC_score)

        log_debug(logger, f"{i}: BIC_score = {BIC_score}")
        if BIC_score > min_BIC_score:
            log_debug(logger, "best_num_fit_GMM = ", i)
            best_num_fit_GMM = i
            break
        min_BIC_score = BIC_score

    return best_num_fit_GMM


# some tests on GMM
# GMM = list of [w(scalar weight), mean(vector), cov(matrix)]
# assume each row of X respresents a point, and the number of rows is the number of points
def BIC(GMM, X, lmbd):
    """Compute Bayesian Information Criterion (BIC) score for a single GMM.
    - https://en.wikipedia.org/wiki/Bayesian_information_criterion

    Args:
        GMM (list): List of [weight, mean, covariance] for each GMM component.
        X (np.ndarray): Feature samples.
        lmbd (float): Regularization parameter.

    Returns:
        float: BIC score.
    """
    dim = GMM[0][1].size

    k = len(GMM) * (1 + dim + (1 + dim) * dim / 2)
    n = X.shape[0]
    epsilon = 1.0e-6
    # sum up log(G(X))
    p = np.zeros(n)
    for g in range(len(GMM)):
        common = 1.0 / np.sqrt(np.linalg.det(2.0 * np.pi * GMM[g][2]))
        mu = np.outer(np.ones(n), GMM[g][1])
        # print("mu.shape: ", mu.shape)
        inv_Cov = np.linalg.inv(GMM[g][2])
        V = np.exp(-0.5 * np.einsum('ij,ij->i', (X - mu) @ inv_Cov, (X - mu)))
        # print("V.shape: ", V.shape)
        p += GMM[g][0] * common * V + epsilon

    return lmbd * k * np.log(n) - 2.0 * np.sum(np.log(p))


# extension to multiple frames
# GMMs = list of GMM
def BICn(GMMs, X, lmbd):
    """Compute BIC score for multiple GMMs (across frames).
    - https://en.wikipedia.org/wiki/Bayesian_information_criterion

    Args:
        GMMs (list[list]): List of GMMs per frame.
        X (np.ndarray): Combined feature samples.
        lmbd (float): Regularization parameter.

    Returns:
        float: BIC score.
    """
    dim = GMMs[0][0][1].size
    # print(f"dim: {dim}")
    k = len(GMMs) * len(GMMs[0]) * (1 + dim + (1 + dim) * dim / 2)
    n = X.shape[0]
    epsilon = 1.0e-6
    # sum up log(G(X))
    p = np.zeros(n)
    for f in range(len(GMMs)):
        for g in range(len(GMMs[f])):
            common = 1.0 / np.sqrt(np.linalg.det(2.0 * np.pi * GMMs[f][g][2]))
            mu = np.outer(np.ones(n), GMMs[f][g][1])
            # print( "mu.shape: ", mu.shape )
            inv_Cov = np.linalg.inv(GMMs[f][g][2])
            V = np.exp(-0.5 * np.einsum('ij,ij->i', (X - mu) @ inv_Cov, (X - mu)))
            # print( "V.shape: ", V.shape )
            p += (GMMs[f][g][0] * common * V + epsilon) / len(GMMs)

    return lmbd * k * np.log(n) - 2.0 * np.sum(np.log(p))


# extension to multiple frames
# GMMs = list of GMM
def BICn_parallel(GMMs, X, lmbd):
    """Compute BIC score for multiple GMMs using parallel processing.
    - https://en.wikipedia.org/wiki/Bayesian_information_criterion

    Args:
        GMMs (list[list]): List of GMMs per frame.
        X (np.ndarray): Combined feature samples.
        lmbd (float): Regularization parameter.

    Returns:
        float: BIC score.
    """
    dim = GMMs[0][0][1].size
    # print(f"dim: {dim}")
    k = len(GMMs) * len(GMMs[0]) * (1 + dim + (1 + dim) * dim / 2)
    n = X.shape[0]
    epsilon = 1.0e-6

    # sum up log(G(X))

    class BICnSubThread(threading.Thread):
        def __init__(self, gmm, X):
            super(BICnSubThread, self).__init__()
            self.gmm = gmm
            self.X = X
            n = X.shape[0]
            self.p = np.zeros(n)

        def run(self):
            for g in range(len(self.gmm)):
                common = 1.0 / np.sqrt(np.linalg.det(2.0 * np.pi * self.gmm[g][2]))
                mu = np.outer(np.ones(n), self.gmm[g][1])
                # print( "mu.shape: ", mu.shape )
                inv_Cov = np.linalg.inv(self.gmm[g][2])
                V = np.exp(-0.5 * np.einsum('ij,ij->i', (self.X - mu) @ inv_Cov, (self.X - mu)))
                # print( "V.shape: ", V.shape )
                self.p += (self.gmm[g][0] * common * V + epsilon)

    p = np.zeros(n)

    num_threads = psutil.cpu_count()
    # print("- num_threads: ", num_threads)

    num_frames = len(GMMs)
    c = 0
    while True:
        if c >= num_frames:
            break

        _num_threads = min([num_threads, num_frames - c])

        threads = []
        for i in range(_num_threads):
            t = BICnSubThread(GMMs[c + i], X)
            t.start()
            threads.append(t)
        for i in range(_num_threads):
            threads[i].join()
            p += threads[i].p / len(GMMs)

        c += _num_threads

    return lmbd * k * np.log(n) - 2.0 * np.sum(np.log(p))


# prepare the above GMM data for integration from fit GMM
def prepare_GMM_set(fit_GMM, feature_samples):
    """Prepare structured GMM data and concatenate all feature samples.

    Args:
        fit_GMM (list[GaussianMixture]): List of fitted GMM models.
        feature_samples (list[np.ndarray]): Feature samples per frame.

    Returns:
        tuple:
            GMM_per_frame (list[list]): List of GMM parameters per frame.
            all_samples (np.ndarray): Concatenated feature samples.
    """
    num_frames = len(fit_GMM)
    num_fit_GMM = fit_GMM[0].n_components

    GMM_per_frame = []

    for f in range(num_frames):
        _per_frame = []
        for g in range(num_fit_GMM):
            _per_frame.append([fit_GMM[f].weights_[g], fit_GMM[f].means_[g], fit_GMM[f].covariances_[g]])
        GMM_per_frame.append(_per_frame)

    all_samples = np.concatenate(feature_samples, axis=0)

    return GMM_per_frame, all_samples


# BIC scores of a single frame
def compute_BIC_of_single_exemplar(GMM_per_frame, all_samples):
    """Compute BIC score for each frame assuming a single exemplar.

    Args:
        GMM_per_frame (list[list]): GMM parameters per frame.
        all_samples (np.ndarray): All concatenated feature samples.

    Returns:
        list[float]: BIC scores for each frame.
    """

    print("# compute_coverages_of_single_exemplar")
    num_frames = len(GMM_per_frame)
    print(f"- num_frames: {num_frames}")

    bic_scores = []
    for f in tqdm.tqdm(range(num_frames)):
        bic_scores.append(BIC(GMM_per_frame[f], all_samples))

    return bic_scores


def compute_BIC_of_n_exemplar(GMM_per_frames, all_samples, num_exemplars=1, parallel=True):
    """Evaluate BIC scores for all combinations of exemplar sets.

    Args:
        GMM_per_frames (list[list]): GMMs for all frames.
        all_samples (np.ndarray): Concatenated feature samples.
        num_exemplars (int): Number of exemplars to choose.
        parallel (bool): Whether to use parallel processing.

    Returns:
        tuple:
            bic_scores (list[float]): List of BIC scores.
            combinations (list[tuple]): List of frame index combinations.
    """
    num_frames = len(GMM_per_frames)
    bic_scores = []

    combinations = list(itertools.combinations(np.arange(num_frames), num_exemplars))
    bic_scores = []
    for fs in combinations:
        bic_scores.append(1.0e33)

    if parallel:
        class BICnThread(threading.Thread):
            def __init__(self, GMM_frames, all_samples):
                super(BICnThread, self).__init__()
                self.GMM_frames = GMM_frames
                self.all_samples = all_samples
                self.score = 1.0e33

            def run(self):
                self.score = BICn(self.GMM_frames, self.all_samples)

        num_threads = psutil.cpu_count()
        log_debug(logger, f"num_threads: {num_threads}")

        num_combinations = len(combinations)
        c = 0
        while True:
            if c >= num_combinations:
                break

            _num_threads = min([num_threads, num_combinations - c])

            threads = []
            for i in range(_num_threads):
                GMM_frames = []
                for j in range(len(combinations[c + i])):
                    GMM_frames.append(GMM_per_frames[combinations[c + i][j]])

                t = BICnThread(GMM_frames, all_samples)
                t.start()
                threads.append(t)
            for i in range(_num_threads):
                threads[i].join()
                bic_scores[c + i] = threads[i].score

            c += _num_threads
    else:
        for k, fs in tqdm.tqdm(enumerate(combinations)):
            GMM_frames = []
            for i in range(len(fs)):
                GMM_frames.append(GMM_per_frames[fs[i]])
            score = BICn(GMM_frames, all_samples)
            # print("score: ", score)
            bic_scores[k] = score

    return bic_scores, combinations


def compute_BIC_of_n_exemplar_greedy(GMM_per_frames, all_samples, min_exemplars_ids, lmbd, parallel=True):
    """Evaluate BIC scores greedily by adding one exemplar at a time.

    Args:
        GMM_per_frames (list[list]): GMMs per frame.
        all_samples (np.ndarray): Concatenated feature samples.
        min_exemplars_ids (list[int]): Current selected exemplar indices.
        lmbd (float): Regularization parameter.
        parallel (bool): Whether to use multi-threading.

    Returns:
        list[float]: BIC scores for each candidate frame.
    """
    num_frames = len(GMM_per_frames)
    # print("- num_frames: ", num_frames)
    # print("- all_samples.shape: ", all_samples.shape)

    bic_scores = []
    for fs in range(num_frames):
        bic_scores.append(1.0e33)

    if parallel:
        class BICnThread(threading.Thread):
            def __init__(self, GMM_frames, all_samples, lmbd):
                super(BICnThread, self).__init__()
                self.GMM_frames = GMM_frames
                self.all_samples = all_samples
                self.lmbd = lmbd
                self.score = 1.0e33

            def run(self):
                self.score = BICn(self.GMM_frames, self.all_samples, self.lmbd)

        num_threads = psutil.cpu_count()
        log_debug(logger, f"num_threads: {num_threads}")

        c = 0
        while True:
            if c >= num_frames:
                break

            _num_threads = min([num_threads, num_frames - c])

            threads = []
            for i in range(_num_threads):
                GMM_frames = []
                for k in min_exemplars_ids:
                    GMM_frames.append(GMM_per_frames[k])
                GMM_frames.append(GMM_per_frames[c + i])

                t = BICnThread(GMM_frames, all_samples, lmbd)
                t.start()
                threads.append(t)
            for i in range(_num_threads):
                threads[i].join()
                bic_scores[c + i] = threads[i].score

            c += _num_threads
    else:
        for k in tqdm.tqdm(range(num_frames)):
            GMM_frames = []
            for i in min_exemplars_ids:
                GMM_frames.append(GMM_per_frames[i])
            GMM_frames.append(GMM_per_frames[k])
            score = BICn(GMM_frames, all_samples, lmbd)
            # print("score: ", score)
            bic_scores[k] = score

    return bic_scores


def estimate_exemplar_frames_greedy(
        feature_file_templates,
        frames=[1],
        max_num_exemplars=None,
        num_fit_GMM=15,
        a=1 / 6,
        num_per_samples_frames=None,
        resolution=None,
        parallel=True,
        plot_dir="plot/exemplar_estimation",
        lmbd=None,
        stop_on_best_exemplars=True,
        plot=False,
        vis_dim0=1,
        vis_dim1=5,
):
    """Estimate optimal exemplar frames using greedy BIC minimization.

    Args:
        feature_file_templates (dict): Mapping of feature name to file template.
        frames (list[int]): Frame indices to consider.
        max_num_exemplars (int): Maximum number of exemplars to select.
        num_fit_GMM (int): Number of GMM components.
        a (float): Scaling factor for lambda.
        num_per_samples_frames (int): Number of features to sample per frame.
        resolution (tuple[int, int]): Resize resolution.
        parallel (bool): Use multi-threading.
        plot_dir (str): Directory to save plots.
        lmbd (float): Lambda regularization value.
        stop_on_best_exemplars (bool): Stop search early if BIC worsens.
        plot (bool): Enable plotting.
        vis_dim0 (int): Feature dimension index for x-axis.
        vis_dim1 (int): Feature dimension index for y-axis.

    Returns:
        tuple:
            min_exemplars (list[int]): Selected exemplar frame indices.
            interpolation_frame (int): Final interpolation frame index.
            lmbd (float): Final lambda value used.
    """
    num_exemplars_list = range(1, max_num_exemplars + 1)
    log_subsection("Feature Sampling")
    feature_samples = sample_features_from_frames(feature_file_templates,
                                                  frames=frames,
                                                  num_per_samples_frames=num_per_samples_frames,
                                                  resolution=resolution)
    _min_bic_score = 1.0e33
    opt_logs = {
        "min_id": [],
        "min_score": [],
        "min_exemplars_ids": [],
        "min_exemplars": []
    }

    if num_fit_GMM is None:
        num_fit_GMM = search_optimal_num_fit_GMM(feature_samples, lmbd, parallel=parallel)

    if lmbd is None:
        K = num_fit_GMM
        d = feature_samples[0].shape[1]

        log_subsection("Determine lambda")
        log_info(logger, f"a: {a}")
        log_info(logger, f"K: {K}")
        log_info(logger, f"d: {d}")

        lmbd = a * (K - 1) * d * (d + 3) / 2

        log_info(logger, f"[λ calculation] lambda: {lmbd:.4f}")

    log_subsection("Fit GMM")
    with Timer(name="fitGMMAll"):

        with Timer(name="fitGMM", level=1) as timer_fitGMM:
            fitted_GMM = fit_GMM_runner(feature_samples, num_fit_GMM, parallel=parallel)

            log_info(logger,
                     f"[GMM fitting] Total: {timer_fitGMM.elapsed():.2f} sec (per frame: {timer_fitGMM.elapsed() / len(feature_samples):.3f} sec)")

        with Timer(name="fitGMMToGMMData", level=1):
            GMM_per_frames = convert_GMMmodels_to_GMMdata(fitted_GMM, num_fit_GMM)

    all_samples = prepare_all_samples(feature_samples)

    min_exemplars_ids = []

    log_subsection("Compute BIC")

    with Timer(name=f"compute_BIC_All"):
        for num_exemplars in num_exemplars_list:
            with Timer(name=f"compute_BIC(num_exemplar={num_exemplars})", level=1):
                bic_scores = compute_BIC_of_n_exemplar_greedy(
                    GMM_per_frames, all_samples, min_exemplars_ids, lmbd,
                    parallel=parallel
                )

                min_id = np.argmin(bic_scores)
                min_score = bic_scores[min_id]

                min_exemplars_ids.append(min_id)

                min_exemplars = [frames[i] for i in min_exemplars_ids]

                log_info(logger, f"[BIC Search: {num_exemplars} exemplar(s)]")
                log_info(logger, f"  -> Best index: {min_id} (score = {min_score:.2f})")
                log_info(logger, f"  -> Exemplar indices: {min_exemplars_ids}, frames: {min_exemplars}")

            opt_logs["min_id"].append(min_id)
            opt_logs["min_score"].append(min_score)
            opt_logs["min_exemplars_ids"].append(min_exemplars_ids)
            opt_logs["min_exemplars"].append(min_exemplars)

            if plot:
                plot_BIC_scores(bic_scores, min_exemplars, num_exemplars, plot_dir)

            if min_score > _min_bic_score and stop_on_best_exemplars:
                log_info(logger, f"Best number of exemplars: {num_exemplars - 1} (no further improvement)")
                break

            _min_bic_score = min_score

    min_exemplars = opt_logs["min_exemplars"][-2]
    min_exemplars_ids = opt_logs["min_exemplars_ids"][-2]
    interpolation_frame = opt_logs["min_exemplars"][-1][-1]

    # if plot:
    #     plot_GMM_density_set(
    #         feature_samples, all_samples,
    #         fitted_GMM, num_fit_GMM,
    #         frames,
    #         min_exemplars_ids,
    #         vis_dim0, vis_dim1,
    #         plot_dir
    #     )

    # if plot:
    #     plot_BIC_scores_for_optimal_exemplars(opt_logs["min_score"], min_exemplars, plot_dir)

    num_exemplars_str = f"{len(min_exemplars)}: {min_exemplars}"

    log_subsection("Final Result")
    log_info(logger, f"[Selected exemplar frames] {num_exemplars_str}")

    return min_exemplars, interpolation_frame, lmbd
