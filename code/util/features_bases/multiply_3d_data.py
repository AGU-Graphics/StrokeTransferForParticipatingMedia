# -----------------------------------------------------------------------------
# Stroke Transfer for Participating Media
# http://cg.it.aoyama.ac.jp/yonghao/sig25/abstsig25.html
#
# File: multiply_3d_data.py
# Maintainer: Naoto Shirashima
#
# Description:
# Integration of 3D data
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
import taichi as ti
import numpy as np
import sys
import os
from data_io import DataIO3D, Output3D

def is_float(value):
    try:
        float(value)
        return True
    except ValueError:
        return False

def main():
    ti.init( arch = ti.gpu )
    
    input_hdf5_file_1 = sys.argv[1]
    input_hdf5_file_2_or_scalar = sys.argv[2]
    output_hdf5_file = sys.argv[3]
    
    input = DataIO3D.InitLoadFile( input_hdf5_file_1, False )
    res = input.get_shape()
    data_1 = input.data_np

    if not is_float( input_hdf5_file_2_or_scalar ):
        input = ( DataIO3D().InitLoadFile( input_hdf5_file_2_or_scalar, False ) )
        print( 'input data2 input file path', input_hdf5_file_2_or_scalar )
        res = input.get_shape()
        print('shape', res)
        data_2 = input.data_np
        
    else:
        data_2 = float( input_hdf5_file_2_or_scalar ) * np.ones_like( data_1 )

    
    output_data = data_1 * data_2
    Output3D( input.bb_min, input.cell_width, output_data, output_hdf5_file )

if __name__ == '__main__':
    main()