# -----------------------------------------------------------------------------
# Stroke Transfer for Participating Media
# http://cg.it.aoyama.ac.jp/yonghao/sig25/abstsig25.html
#
# File: util/exemplar_estimation/plot.py
# Maintainer: Hideki Todo
#
# Description:
# Visualizes GMM densities and BIC scores for exemplar frame selection in feature space.
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

import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import multivariate_normal
from sklearn.mixture import GaussianMixture

from util.infra.logger import getLogger, log_subsection, log_debug

logger = getLogger()


def plot_GMM_density(
        gmm_or_list,
        dim0=0,
        dim1=1,
        out_file="gmm_density.png",
        resolution=(512, 512),
        x_range=None,
        y_range=None,
        normalize=False,
):
    """Plot 2D density map from one or more GMM models and save it as an image.

    Args:
        gmm_or_list (GaussianMixture or list[GaussianMixture]): One or more fitted GMM models.
        dim0 (int): Index of the feature dimension for x-axis.
        dim1 (int): Index of the feature dimension for y-axis.
        out_file (str): Path to output image file.
        resolution (tuple[int, int]): Output resolution (width, height).
        x_range (tuple[float, float], optional): x-axis value range.
        y_range (tuple[float, float], optional): y-axis value range.
        normalize (bool): Whether to normalize and invert the density values for contrast.
    """

    if not isinstance(gmm_or_list, list):
        gmm_list = [gmm_or_list]
    else:
        gmm_list = gmm_or_list

    if x_range is None or y_range is None:
        all_means = np.vstack([gmm.means_[:, [dim0, dim1]] for gmm in gmm_list])
        min_xy = all_means.min(axis=0)
        max_xy = all_means.max(axis=0)
        x_range = (min_xy[0], max_xy[0])
        y_range = (min_xy[1], max_xy[1])

    x_min, x_max = x_range
    y_min, y_max = y_range

    w, h = resolution
    x = np.linspace(x_min, x_max, w)
    y = np.linspace(y_min, y_max, h)
    X, Y = np.meshgrid(x, y)
    XY = np.stack([X, Y], axis=-1)

    Z = np.zeros((h, w), dtype=np.float64)

    for gmm in gmm_list:
        means = gmm.means_[:, [dim0, dim1]]
        covariances = []
        for cov in gmm.covariances_:
            if cov.ndim == 2:
                cov_2d = cov[np.ix_([dim0, dim1], [dim0, dim1])]
            elif cov.ndim == 1:
                cov_2d = np.diag([cov[dim0], cov[dim1]])
            else:
                raise ValueError(f"Unsupported covariance shape: {cov.shape}")
            covariances.append(cov_2d)

        for weight, mean, cov in zip(gmm.weights_, means, covariances):
            try:
                rv = multivariate_normal(mean, cov, allow_singular=True)
                Z += weight * rv.pdf(XY)
            except np.linalg.LinAlgError:
                continue

    if normalize:
        Z = Z - Z.min()
        Z = Z / (Z.max() + 1e-9)
        Z = 1.0 - Z
    else:
        Z = np.clip(1.0 - Z / Z.max(), 0, 1)

    plt.imsave(out_file, Z, cmap="gray", vmin=0, vmax=1)


def plot_GMM_density_set(
        feature_samples,
        all_samples,
        fit_GMM,
        num_fit_GMM,
        frames,
        min_exemplars_ids,
        vis_dim0, vis_dim1,
        plot_dir
):
    """Plot density maps for each frame, all frames, and selected exemplars.

    Args:
        feature_samples (list[np.ndarray]): Feature samples per frame.
        all_samples (np.ndarray): All feature samples concatenated.
        fit_GMM (list[GaussianMixture]): List of per-frame GMM models.
        num_fit_GMM (int): Number of components per GMM.
        frames (list[int]): List of frame indices.
        min_exemplars_ids (list[int]): Indices of selected exemplar frames.
        vis_dim0 (int): Feature dimension index for x-axis.
        vis_dim1 (int): Feature dimension index for y-axis.
        plot_dir (str): Directory where plots will be saved.
    """
    all_xy = all_samples[:, [vis_dim0, vis_dim1]]
    xy_mean = np.mean(all_xy, axis=0)
    xy_std = np.std(all_xy, axis=0)

    k_std = 2.0

    x_min, x_max = xy_mean[0] - k_std * xy_std[0], xy_mean[0] + k_std * xy_std[0]
    y_min, y_max = xy_mean[1] - k_std * xy_std[1], xy_mean[1] + k_std * xy_std[1]

    x_range = (x_min, x_max)
    y_range = (y_min, y_max)

    log_subsection("Visualize frame GMM densities")

    out_gmm_vis_dir = os.path.join(plot_dir, "gmm_density")
    os.makedirs(out_gmm_vis_dir, exist_ok=True)

    for i, gm in enumerate(fit_GMM):
        out_path = os.path.join(out_gmm_vis_dir, f"gmm_frame_{frames[i]:03d}.png")
        plot_GMM_density(gm, dim0=vis_dim0, dim1=vis_dim1, out_file=out_path,
                         x_range=x_range, y_range=y_range)

    log_subsection("Visualize all-frame GMM densities")

    gmm_all = GaussianMixture(n_components=num_fit_GMM, random_state=0).fit(all_samples)
    out_path_all = os.path.join(plot_dir, "gmm_all.png")
    plot_GMM_density(gmm_all, dim0=vis_dim0, dim1=vis_dim1, out_file=out_path_all,
                     x_range=x_range, y_range=y_range)

    log_subsection("Visualize Selected GMM")
    log_debug(logger, f"min_exemplars_ids: {min_exemplars_ids}")

    selected_gmm = [fit_GMM[i] for i in min_exemplars_ids]
    selected_samples = [feature_samples[i] for i in min_exemplars_ids]

    out_path_sel = os.path.join(plot_dir, "gmm_selected.png")

    all_selected_samples = np.concatenate(selected_samples, axis=0)
    gmm_selected = GaussianMixture(n_components=num_fit_GMM, random_state=0).fit(all_selected_samples)
    plot_GMM_density(gmm_selected, dim0=vis_dim0, dim1=vis_dim1, out_file=out_path_sel,
                     x_range=x_range, y_range=y_range)


def plot_BIC_scores(bic_scores, min_exemplars, num_exemplars, plot_dir):
    """Plot and save BIC scores across frame indices for fixed exemplar count.

    Args:
        bic_scores (list[float]): BIC scores per candidate frame.
        min_exemplars (list[int]): Selected exemplar frame indices.
        num_exemplars (int): Number of exemplars selected.
        plot_dir (str): Directory where plot will be saved.
    """
    min_id = np.argmin(bic_scores)
    min_score = bic_scores[min_id]

    fig = plt.figure(figsize=(12, 6))
    plt.plot(bic_scores, "o-")
    plt.title(f"min_idx, score, exemplars = {min_id, min_score, min_exemplars}")

    out_file = f"{plot_dir}/find_best_num_exemplars_n{num_exemplars}.png"
    out_dir = os.path.dirname(out_file)
    os.makedirs(out_dir, exist_ok=True)

    fig.savefig(out_file, bbox_inches="tight", pad_inches=0.05, transparent=False)


def plot_BIC_scores_for_optimal_exemplars(min_scores, min_exemplars, plot_dir):
    """Plot and save BIC scores during greedy selection process.

    Args:
        min_scores (list[float]): Minimum BIC scores for increasing number of exemplars.
        min_exemplars (list[int]): Final set of selected exemplar indices.
        plot_dir (str): Directory where plot will be saved.
    """
    num_exemplars = len(min_exemplars)
    fig = plt.figure(figsize=(12, 6))
    plt.plot(min_scores, "o-")
    plt.title(f"Best: n={num_exemplars}, " + str(min_exemplars))
    out_file = f"{plot_dir}/best_num_exemplars_n{num_exemplars}.png"
    out_dir = os.path.dirname(out_file)
    os.makedirs(out_dir, exist_ok=True)

    fig.savefig(out_file, bbox_inches="tight", pad_inches=0.05, transparent=False)
