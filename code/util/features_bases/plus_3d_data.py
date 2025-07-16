# -----------------------------------------------------------------------------
# Stroke Transfer for Participating Media
# http://cg.it.aoyama.ac.jp/yonghao/sig25/abstsig25.html
#
# File: plus_3d_data.py
# Maintainer: Naoto Shirashima
#
# Description:
# Adding 3D data together
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
import sys
import h5py
from data_io import DataIO3D, Output3D


def main():
    input_hdft_file = sys.argv[1]
    data_io = DataIO3D.InitLoadFile(input_hdft_file, taichi=False)
    data_1 = data_io.data_np

    input_hdft_file = sys.argv[2]
    data_2  = DataIO3D.InitLoadFile(input_hdft_file, taichi=False).data_np

    output_hdf5_file = sys.argv[3]
    output_data = data_1 + data_2

    bb_min = data_io.bb_min
    cell_width = data_io.cell_width
    
    Output3D(bb_min, cell_width, output_data, output_hdf5_file)

if __name__ == '__main__':
    main()