# -----------------------------------------------------------------------------
# Stroke Transfer for Participating Media
# http://cg.it.aoyama.ac.jp/yonghao/sig25/abstsig25.html
#
# File: gradient_feature.py
# Maintainer: Naoto Shirashima
#
# Description:
# Compute gradients from scalar features
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
import matplotlib.pyplot as plt
import numpy as np
import taichi as ti
import taichi.math as tm
import sys
from data_io import DataIO2D, Output2D
from interpolation_taichi_field import CubicBSpline2D


def main():
    compute_mode = sys.argv[1]
    if compute_mode == 'cpu':
        ti.init(arch=ti.cpu)
    elif compute_mode == 'gpu':
        ti.init(arch=ti.gpu)

    input_file_path = sys.argv[2]
    output_file_path = sys.argv[3]
    h5 = DataIO2D.InitLoadFile(input_file_path, taichi=False)
    data_np = h5.data_np.swapaxes(0, 1)
    shape = data_np.shape

    spline = CubicBSpline2D(data_np, [DataIO2D.screen_bb_min_0, DataIO2D.screen_bb_min_1], DataIO2D.screen_cell_width)

    gradientScreen = spline.GradientArray(shape).to_numpy()
    
    gradientScreen = gradientScreen.swapaxes(0, 1)
    Output2D(gradientScreen, output_file_path)

if __name__ == '__main__':
    main()
