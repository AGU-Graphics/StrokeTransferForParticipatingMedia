# -----------------------------------------------------------------------------
# Stroke Transfer for Participating Media
# http://cg.it.aoyama.ac.jp/yonghao/sig25/abstsig25.html
#
# File: convert_to_transmittance_cell.py
# Maintainer: Naoto Shirashima
#
# Description:
# Conversion of accumulated extinction coefficients over a cell as seen through a screen into transmittance.
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
import matplotlib.pyplot as plt
import sys
from data_io import DataIO3D, Output3D

def main():
    input_h5 = DataIO3D.InitLoadFile(sys.argv[1], taichi=False)
    res = input_h5.get_shape()
    data_1 = input_h5.data_np

    output_hdf5_file = sys.argv[2]
    output_data = np.exp(-data_1)

    Output3D(input_h5.bb_min, input_h5.cell_width, output_data, output_hdf5_file)

if __name__ == '__main__':
    main()