# -----------------------------------------------------------------------------
# Stroke Transfer for Participating Media
# http://cg.it.aoyama.ac.jp/yonghao/sig25/abstsig25.html
#
# File: convert_to_transmittance_screen.py
# Maintainer: Naoto Shirashima
#
# Description:
# Conversion of accumulated extinction coefficients on a screen to transmittance as seen from the screen
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
from data_io import DataIO2D, Output2D

def main():
    input_hdf5_file = sys.argv[1]
    data_1 = DataIO2D.InitLoadFile(input_hdf5_file, taichi=False).data_np

    output_hdf5_file = sys.argv[2]
    output_data = np.exp(-data_1)
    Output2D( output_data, output_hdf5_file)

if __name__ == '__main__':
    main()