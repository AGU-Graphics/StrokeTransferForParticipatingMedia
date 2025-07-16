# -----------------------------------------------------------------------------
# Stroke Transfer for Participating Media
# http://cg.it.aoyama.ac.jp/yonghao/sig25/abstsig25.html
#
# File: normalize_and_rotate_basis.py
# Maintainer: Naoto Shirashima
#
# Description:
# Generate normalized vectors and normalized rotation vectors for gradient data
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
import argparse
import sys
import numpy as np
import taichi as ti
import taichi.math as tm
from data_io import DataIO2D, Output2D

args = argparse.ArgumentParser()
# args.add_argument('-dim', type=int, choices=[2, 3], required=True,help='input data dimension (2 or 3)')
args.add_argument('-i', '--inputs', nargs='+', required=True, help='Input HDF5 file(s): either a single file or one file per dimension.')
args.add_argument('-compute_mode', type=str, choices=['cpu', 'gpu'], default='gpu', help='Compute mode: "cpu" or "gpu". Default is "gpu".')
args.add_argument('-out_norm', default=None, help='output norm vector')
args.add_argument('-out_perp', required=True, help='output perpendicular vector')
args.add_argument('-out_para', required=True, help='output parallel vector')
args = args.parse_args()


dim = 2
input_files = args.inputs
one_file_mode = len(input_files) == 1
split_file_mode = not one_file_mode

if one_file_mode:
    h5 = DataIO2D.InitLoadFile(input_files[0], taichi=False)
    data_np = h5.data_np[:, :, :dim]
else:
    split = [DataIO2D.InitLoadFile(f, taichi=False).data_np for f in input_files]
    data_np = np.stack(split, axis=-1)         # (H, W, dim)

H, W, _ = data_np.shape

# ti.init(arch=ti.gpu if ti.detect_gpu_arch() else ti.cpu)
if args.compute_mode == 'gpu':
    ti.init(arch=ti.gpu)
else:
    ti.init(arch=ti.cpu)

input_data = ti.Vector.field(dim, dtype=ti.f32, shape=(H, W))
output_norm = ti.field(dtype=ti.f32, shape=(H, W))
output_perp = ti.Vector.field(dim, dtype=ti.f32, shape=(H, W))
output_para = ti.Vector.field(dim, dtype=ti.f32, shape=(H, W))

input_data.from_numpy(data_np)

@ti.kernel
def normalize_k(dim: ti.i32):
    for u, v in ti.ndrange(H, W):
        n = ti.sqrt(input_data[u, v][0]**2 + input_data[u, v][1]**2)
        output_norm[u, v] = n

        if n > 1e-5:
            inv = 1.0 / n
            output_perp[u, v] = tm.vec2([input_data[u, v][0] * inv,
                                        input_data[u, v][1] * inv])
            output_para[u, v] = tm.vec2([ -output_perp[u, v][1], output_perp[u, v][0] ])
        else:
            output_perp[u, v] = tm.vec2([0.0, 0.0])
            output_para[u, v] = tm.vec2([0.0, 0.0])

normalize_k(dim)

output_norm = output_norm.to_numpy()
output_perp = output_perp.to_numpy()
output_para = output_para.to_numpy()

if args.out_norm is not None:
    Output2D( output_norm, args.out_norm )
Output2D( output_perp, args.out_perp )
Output2D( output_para, args.out_para )
