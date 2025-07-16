# -----------------------------------------------------------------------------
# Stroke Transfer for Participating Media
# http://cg.it.aoyama.ac.jp/yonghao/sig25/abstsig25.html
#
# File: temperature_scaler.py
# Maintainer: Naoto Shirashima
#
# Description:
# Convert Houdini temperature field to Kelvin
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
import os
from data_io import DataIO2D, Output2D
import argparse

def main():
    args = argparse.ArgumentParser()
    args.add_argument('-i', type=str, required=True, help='Temperature 2D data file path')
    args.add_argument('-min', type=float, required=True, help='Minimum temperature value')
    args.add_argument('-max', type=float, required=True, help='Maximum temperature value')
    args.add_argument('-o', type=str, required=True, help='Output HDF5 file path')
    args = args.parse_args()
    
    input_hdft_file = args.i
    data_1 = DataIO2D.InitLoadFile(input_hdft_file, taichi=False).data_np

    min_temp = args.min
    max_temp = args.max

    output_hdf5_file = args.o
    output_data = data_1 * (max_temp - min_temp) + min_temp

    Output2D(output_data, output_hdf5_file)

if __name__ == '__main__':
    main()