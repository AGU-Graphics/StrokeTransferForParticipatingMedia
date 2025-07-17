# -----------------------------------------------------------------------------
# Stroke Transfer for Participating Media
# http://cg.it.aoyama.ac.jp/yonghao/sig25/abstsig25.html
#
# File: black_body.py
# Maintainer: Naoto Shirashima and Yuki Yamaoka
#
# Description:
# Converting temperature field to brightness
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
import os
from data_io import DataIO3D
from interpolation_taichi_field import InterpolationField

class BlackBodyField:
    def __init__(self, min_temp, max_temp, temp_factor, blackbody_factor, hdf5_file_path):
        self.min_temp = min_temp
        self.max_temp = max_temp
        self.temp_factor = temp_factor
        self.blackbody_factor = blackbody_factor

        if hdf5_file_path == '' or hdf5_file_path == None:
            self.bb_min = tm.vec3([0.,0.,0.])
            self.bb_max = tm.vec3([0.,0.,0.])
            self.cell_width = tm.vec3([0.,0.,0.])
            self.shape = ti.field(ti.i32, shape=(3))
            self.data = ti.field(ti.f32, shape=(1,1,1))
            self.max_temp = 0.0
        else:
            input_h5 = DataIO3D.InitLoadFile(hdf5_file_path)
            self.bb_min = input_h5.bb_min
            self.bb_max = input_h5.bb_max_taichi()
            self.cell_width = input_h5.cell_width
            self.shape = input_h5.get_shape()
            self.data_np = input_h5.data_np
            self.data_np = self.data_np * self.temp_factor
            self.data = input_h5.data

    @ti.func
    def GetSpectralRadiance(self, position: tm.vec3, wavelength: float):
        result = 0.0
        if self.max_temp > 0.0:
            bb_min = self.bb_min
            bb_max = self.bb_max
            cell_width = self.cell_width
            PLANCK = 6.62607015e-34       # [J / Hz]
            BOLTZMANN = 1.380658e-23      # [J / K]
            SPEED_OF_LIGHT = 2.99792458e8  # [m / s]
            T = InterpolationField(bb_min, bb_max, self.data, position, cell_width) * (self.max_temp - self.min_temp) + self.min_temp
            lamb = wavelength

            radiance = 2.0 * PLANCK * SPEED_OF_LIGHT**2 / lamb**5 / (tm.exp(PLANCK * SPEED_OF_LIGHT / BOLTZMANN / T / lamb) - 1.0)
            result = radiance * self.blackbody_factor
        return result
    
    @ti.func
    def GetSpectralRadianceVec3(self, position: tm.vec3, wavelength: tm.vec3):
        result = tm.vec3([0.0, 0.0, 0.0])
        if self.max_temp > 0.0:
            bb_min = self.bb_min
            bb_max = self.bb_max
            cell_width = self.cell_width
            T = InterpolationField(bb_min, bb_max, self.data, position, cell_width) * (self.max_temp - self.min_temp) + self.min_temp
            PLANCK = 6.62607015e-34       # [J / Hz]
            BOLTZMANN = 1.380658e-23      # [J / K]
            SPEED_OF_LIGHT = 2.99792458e8  # [m / s]

            for i in ti.static(range(3)):
                lamb = wavelength[i]
                radiance = 2.0 * PLANCK * SPEED_OF_LIGHT**2 / lamb**5 / (tm.exp(PLANCK * SPEED_OF_LIGHT / BOLTZMANN / T / lamb) - 1.0)
                result[i] = radiance * self.blackbody_factor
        return result