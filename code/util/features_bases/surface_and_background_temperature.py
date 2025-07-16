# -----------------------------------------------------------------------------
# Stroke Transfer for Participating Media
# http://cg.it.aoyama.ac.jp/yonghao/sig25/abstsig25.html
#
# File: surface_and_background_temperature.py
# Maintainer: Naoto Shirashima
#
# Description:
# Generate temperature fields for surface models and backgrounds
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
import argparse
from data_io import DataIO2D, Output2D

def main():
    args = argparse.ArgumentParser()
    args.add_argument('-depth', type=str, required=True, help='Depth value file path')
    args.add_argument('-back', '--background_temperature', type=float, required=True, help='Background temperature value')
    args.add_argument('-surf', '--surface_temperature', type=float, required=True, help='Surface temperature value')
    args.add_argument('-out', '--output_hdf5_file', type=str, required=True, help='Output HDF5 file path')
    args = args.parse_args()
    data_1 = DataIO2D.InitLoadFile(args.depth, taichi=False).data_np

    background_temperature = args.background_temperature
    surface_temperature = args.surface_temperature
    output_hdf5_file = args.output_hdf5_file

    out_temp = data_1
    out_temp[data_1 < np.inf] = surface_temperature
    out_temp[data_1 == np.inf] = background_temperature

    Output2D(out_temp, output_hdf5_file)

if __name__ == '__main__':
    main()