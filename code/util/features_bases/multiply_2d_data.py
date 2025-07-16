# -----------------------------------------------------------------------------
# Stroke Transfer for Participating Media
# http://cg.it.aoyama.ac.jp/yonghao/sig25/abstsig25.html
#
# File: multiply_2d_data.py
# Maintainer: Naoto Shirashima
#
# Description:
# Integration of 2D data
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
import cv2
from data_io import DataIO2D, Output2D

def is_float(str):
    try:
        float(str)
        return True
    except ValueError:
        return False

def main():
    input_hdf5_file = sys.argv[1]
    data_1 = DataIO2D.InitLoadFile(input_hdf5_file, taichi=False).data_np
    res1 = data_1.shape

    input_hdf5_file = sys.argv[2]
    if not is_float(input_hdf5_file):
        data_2 = DataIO2D.InitLoadFile(input_hdf5_file, taichi=False).data_np
        data_2 = cv2.resize(data_2, (res1[1], res1[0]), interpolation=cv2.INTER_LINEAR)
        data_2 = np.array(data_2)
    else:
        print(input_hdf5_file, float(input_hdf5_file))
        data_2 = np.ones_like(data_1) * float(input_hdf5_file)

    output_hdf5_file = sys.argv[3]
    output_data = data_1 * data_2

    Output2D(output_data, output_hdf5_file)

if __name__ == '__main__':
    main()