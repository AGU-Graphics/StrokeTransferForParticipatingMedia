# -----------------------------------------------------------------------------
# Stroke Transfer for Participating Media
# http://cg.it.aoyama.ac.jp/yonghao/sig25/abstsig25.html
#
# File: util/common/feature_basis_def.py
# Maintainer: Hideki Todo
#
# Description:
# This module defines raw and processed feature names, basis vector categories, and symbolic labels for visualization.
# It serves as a centralized definition for feature naming and symbolic mappings used in our pipelines.
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
# ========= Raw Feature Names =========
# Raw input features before standardization
RAW_FEATURE_NAMES = [
    # Transmittance
    "transmittance",

    # Intensity and Intensity Gradient
    "intensity_r",
    "intensity_g",
    "intensity_b",
    "intensity_l",
    "intensity_l_gradient",

    # Silhouette Distance
    "silhouette_distance",

    # Apparent Normals
    "apparent_normal_x",
    "apparent_normal_y",
    "apparent_normal_z",

    # Apparent Relative Velocity
    "apparent_relative_velocity_u",
    "apparent_relative_velocity_v",
    "apparent_relative_velocity_norm",

    # Gaussian and Mean Curvatures
    "gaussian_curvature",
    "mean_curvature",

    # Mean-Free-Path
    "mean_free_path",

    # Temperature
    "temperature",

    # Bounding Box
    "bounding_box"
]

# ========= Feature Names =========
# Processed features used in regression and transfer
FEATURE_NAMES = [
    # Transmittance
    "transmittance",

    # Intensity and Intensity Gradient (LAB)
    "intensity_l",
    "intensity_astar",
    "intensity_bstar",
    "intensity_l_gradient",

    # Silhouette Distance
    "silhouette_distance",

    # Apparent Normals
    "apparent_normal_x",
    "apparent_normal_y",
    "apparent_normal_z",

    # Apparent Relative Velocity
    "apparent_relative_velocity_u",
    "apparent_relative_velocity_v",
    "apparent_relative_velocity_norm",

    # Gaussian and Mean Curvatures
    "gaussian_curvature",
    "mean_curvature",

    # Mean-Free-Path
    "mean_free_path",

    # Temperature
    "temperature"
]

# ========== Basis Names ==========
# Category-wise basis vectors (para/perp), shared by basis and basis_smooth
BASIS_NAMES = [
    # Silhouette Guided Basis
    "silhouette_guided_para",
    "silhouette_guided_perp",

    # Apparent Normal Basis
    "apparent_normal_para",
    "apparent_normal_perp",

    # Intensity Gradient Basis
    "intensity_gradient_para",
    "intensity_gradient_perp",

    # Apparent Relative Velocity Basis
    "apparent_relative_velocity_para",
    "apparent_relative_velocity_perp",

    # Gradient of Apparent Mean Free Path Basis
    "mean_free_path_gradient_para",
    "mean_free_path_gradient_perp"
]

# ========= Feature and Basis Symbols =========
# Mapping from feature/basis names to LaTeX-style symbols
FEATURE_BASIS_SYMBOLS = {
    # Features
    "transmittance": "$T_s$",
    "intensity_r": "$r$",
    "intensity_g": "$g$",
    "intensity_b": "$b$",
    "intensity_l": "$I_s$",
    "intensity_astar": "$a^{\\star}_s$",
    "intensity_bstar": "$b^{\\star}_s$",
    "intensity_l_gradient": "$I^{\\nabla_2}_s$",
    "silhouette_distance": "$\\xi_s$",
    "apparent_normal_x": "$n_s^{(x)}$",
    "apparent_normal_y": "$n_s^{(y)}$",
    "apparent_normal_z": "$n_s^{(z)}$",
    "apparent_relative_velocity_u": "$v_s^{(x)}$",
    "apparent_relative_velocity_v": "$v_s^{(y)}$",
    "apparent_relative_velocity_norm": "$|v_s|$",
    "gaussian_curvature": "$\\kappa_{G_s}$",
    "mean_curvature": "$\\kappa_{m_s}$",
    "mean_free_path": "$d_s$",
    "temperature": "$C_s$",

    # Bases
    "silhouette_guided_para": "$\\boldsymbol{o}^{(\\parallel)}$",
    "silhouette_guided_perp": "$\\boldsymbol{o}^{(\\perp)}$",
    "apparent_normal_para": "$\\boldsymbol{n}^{(\\parallel)}$",
    "apparent_normal_perp": "$\\boldsymbol{n}^{(\\perp)}$",
    "intensity_gradient_para": "$\\boldsymbol{I}^{(\\parallel)}$",
    "intensity_gradient_perp": "$\\boldsymbol{I}^{(\\perp)}$",
    "apparent_relative_velocity_para": "$\\boldsymbol{v}^{(\\parallel)}$",
    "apparent_relative_velocity_perp": "$\\boldsymbol{v}^{(\\perp)}$",
    "mean_free_path_gradient_para": "$\\boldsymbol{m}^{(\\parallel)}$",
    "mean_free_path_gradient_perp": "$\\boldsymbol{m}^{(\\perp)}$"
}

# ========== Debug Print ==========
if __name__ == "__main__":
    def print_list(name, values):
        print(f"{name}:")
        for v in values:
            print(f" - {v}")
        print()

    print_list("RAW_FEATURE_NAMES", RAW_FEATURE_NAMES)
    print_list("FEATURE_NAMES", FEATURE_NAMES)
    print_list("BASIS_NAMES", BASIS_NAMES)