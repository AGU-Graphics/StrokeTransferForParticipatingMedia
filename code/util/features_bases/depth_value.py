# -----------------------------------------------------------------------------
# Stroke Transfer for Participating Media
# http://cg.it.aoyama.ac.jp/yonghao/sig25/abstsig25.html
#
# File: depth_value.py
# Maintainer: Naoto Shirashima
#
# Description:
# Depth Value Class
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
import taichi.math as tm
import numpy as np
from data_io import DataIO2D
from sample_points_generator import vu_point_to_uv_point_with_haltonseq
from interpolation_taichi_field import InterpolationField2D

class DepthValue:
    def __init__(self, depth_file_path):
        self.depth_file_path = depth_file_path
        if depth_file_path is None:
            self.depth_file_path = ''
        if self.depth_file_path == '':
            self.depth_data_ti = ti.field(dtype=ti.f32, shape=(1,1))
            print('depth test not available')
        else:
            # self.depth_h5 = h5py.File(self.depth_file_path, 'r')
            # self.resolution = np.array(self.depth_h5['resolution']).flatten()
            dataio_2d = DataIO2D.InitLoadFile(self.depth_file_path)
            self.data = dataio_2d.data_np
            # self.data = np.array(self.depth_h5['data']).reshape((self.resolution[1], self.resolution[0]))
            # self.data = self.data.swapaxes(0, 1)
            # self.depth_data_ti = ti.field(dtype=ti.f32, shape=(self.resolution[0], self.resolution[1]))
            # self.depth_data_ti.from_numpy(depth_data_np)
            self.depth_data_ti = dataio_2d.data
            self.node_num = self.data.shape
            self.bb_max = dataio_2d.bb_max

            print("depth_data shape", self.data.shape)
            print("depth_max", np.max(self.data))
            print("depth_min", np.min(self.data))
            print("depth_sum", np.sum(self.data))

            # self.testdata()
    @ti.func
    def is_valid(self):
        return not self.depth_file_path == ''

    @ti.func
    def InterpolationDepth(self, vu: tm.vec2, super_samples: int):
        uv = vu_point_to_uv_point_with_haltonseq(tm.vec2([vu[0], vu[1]]), super_samples)
        vu = tm.vec2([uv[1], uv[0]])
        bb_min = tm.vec2([DataIO2D.screen_bb_min_0, DataIO2D.screen_bb_min_1])
        bb_cell_length = DataIO2D.screen_cell_width * tm.vec2([1,1])
        return InterpolationField2D(bb_min , self.depth_data_ti, vu, bb_cell_length, clamp_mode=2)

    @ti.func
    def GetDepth01(self, vu_01: tm.vec2):
        resolution = tm.vec2([self.depth_data_ti.shape[0] - 1, self.depth_data_ti.shape[1] - 1])
        vu = vu_01 * resolution
        vu = tm.ivec2([vu[0], vu[1]])
        vu = tm.clamp(vu, 0, resolution - 1)
        return self.depth_data_ti[int(vu[0]), int(vu[1])]

    @ti.func
    def InterpolationDepth01(self, vu_01: tm.vec2, super_samples: int):
        resolution = tm.vec2([self.depth_data_ti.shape[0] - 1, self.depth_data_ti.shape[1] - 1])
        vu = vu_01 * resolution
        return self.InterpolationDepth(vu, super_samples)
