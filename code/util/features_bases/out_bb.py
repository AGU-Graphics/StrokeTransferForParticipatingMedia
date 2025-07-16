# -----------------------------------------------------------------------------
# Stroke Transfer for Participating Media
# http://cg.it.aoyama.ac.jp/yonghao/sig25/abstsig25.html
#
# File: out_bb.py
# Maintainer: Naoto Shirashima
#
# Description:
# Output of BoundingBox information
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
from data_io import DataIO3D
import sys
import h5py
import taichi as ti

def main():
    ti.init(arch=ti.gpu)
    
    input_hdf5 = sys.argv[1]
    out_hdf5 = sys.argv[2]
    
    
    input = DataIO3D.InitLoadFile(input_hdf5, False)
    
    out_h5 = h5py.File(out_hdf5, 'w')
    out_h5.create_dataset('bb_min', data=input.bb_min)
    out_h5.create_dataset('bb_max', data=input.bb_max_python())
    out_h5.close()


if __name__ == '__main__':
    main()