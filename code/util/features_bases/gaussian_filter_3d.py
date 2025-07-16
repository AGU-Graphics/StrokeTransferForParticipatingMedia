# -----------------------------------------------------------------------------
# Stroke Transfer for Participating Media
# http://cg.it.aoyama.ac.jp/yonghao/sig25/abstsig25.html
#
# File: gaussian_filter_3d.py
# Maintainer: Naoto Shirashima
#
# Description:
# Gaussian filter for volume data to calculate curvature
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
import h5py
import os

from mpl_toolkits.mplot3d import Axes3D

from scipy.ndimage import gaussian_filter

from data_io import DataIO3D, Output3D

# 3次元ボリュームのGaussian Filterの実装．
def gaussian_filter_3d(D, sigma, dx, dy, dz):
    print('sigma',sigma, [sigma / dx, sigma / dy, sigma / dz])
    return gaussian_filter(D, sigma=[sigma / dx, sigma / dy, sigma / dz])

def gaussian_3d_grid(data, num_nodes, min_point, max_point, sigma):
    dx = (max_point[0] - min_point[0]) / num_nodes[0]
    dy = (max_point[1] - min_point[1]) / num_nodes[1]
    dz = (max_point[2] - min_point[2]) / num_nodes[2]
    sigma = (max_point - min_point)[0] * sigma
    print('all x sigma', sigma)
    print('dx, dy, dz:', dx, dy, dz)
    B = gaussian_filter_3d(data, sigma, dx, dy, dz)
    return B

def main():
    import sys
    hdf5_file_path = sys.argv[1]
    gaussian_sigma_in_single_cell = float(sys.argv[2])
    output_hdf5_file_path = f'{sys.argv[3]}'

    input_h5 = DataIO3D.InitLoadFile(hdf5_file_path, False)
    min_point = input_h5.bb_min
    max_point = input_h5.bb_max_taichi()
    cell_size = input_h5.cell_width
    num_nodes = input_h5.get_shape()
    data = input_h5.data_np

    gaussian_filtered_data = gaussian_3d_grid(data, num_nodes, min_point, max_point, gaussian_sigma_in_single_cell)
    Output3D(min_point, cell_size, gaussian_filtered_data, output_hdf5_file_path)


if __name__ == '__main__':
    main()