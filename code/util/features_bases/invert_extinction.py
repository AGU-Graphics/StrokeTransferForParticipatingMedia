# -----------------------------------------------------------------------------
# Stroke Transfer for Participating Media
# http://cg.it.aoyama.ac.jp/yonghao/sig25/abstsig25.html
#
# File: invert_extinction.py
# Maintainer: Naoto Shirashima and Yuki Yamaoka
#
# Description:
# The high market extinction coefficient is inverted and converted to a free path distribution on the lattice
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
import argparse
import h5py
import os
import taichi as ti
from data_io import DataIO3D, Output3D

def main():
    arg = argparse.ArgumentParser()
    arg.add_argument('--extinction', type=str, required=True, help='input extinction hdf5 file')
    arg.add_argument('--epsilon', type=float, default=1/2000, help='input epsilon')
    arg.add_argument('--compute_mode', type=str, default='gpu', help='compute mode (cpu or gpu)')
    arg.add_argument('-o', '--output', type=str, required=True, help='output hdf5 file')
    arg = arg.parse_args()
    compute_mode = arg.compute_mode
    extinction_path = arg.extinction
    output = arg.output
    epsilon = arg.epsilon
    
    if compute_mode == 'gpu':
        ti.init(arch=ti.gpu)
    else:
        ti.init(arch=ti.cpu)
    input = DataIO3D().InitLoadFile(extinction_path, taichi=True)
    output_taichi = DataIO3D.TaichiFieldInit(input.get_shape())
    @ti.kernel
    def compute():
        for I in ti.grouped(output_taichi):
            sigma = input.data[I]
            output_taichi[I] = 1.0 / (sigma + epsilon)
    
    compute()
    result_np = output_taichi.to_numpy()
    Output3D(input.bb_min, input.cell_width, result_np, output)
    
    
    

if __name__ == '__main__':
    main()