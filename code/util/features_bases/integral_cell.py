# -----------------------------------------------------------------------------
# Stroke Transfer for Participating Media
# http://cg.it.aoyama.ac.jp/yonghao/sig25/abstsig25.html
#
# File: integral_cell.py
# Maintainer: Naoto Shirashima and Yuki Yamaoka
#
# Description:
# Calculate the extinction coefficient on the grid and the cumulative extinction coefficient on the grid from the camera.
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
import sys
from xml_load import xml_load
from data_io import DataIO3D, Output3D
from bounding_box import GetPositionFromDataIndex, IntersectBB
from interpolation_taichi_field import InterpolationField

@ti.data_oriented
class Integral3D:
    def __init__(self, density_file_path, integral_start):
        input_h5 = DataIO3D.InitLoadFile(density_file_path)

        self.bb_min = input_h5.bb_min
        self.bb_max = input_h5.bb_max_taichi()
        self.cell_width = input_h5.cell_width
        self.shape = input_h5.get_shape()
        self.data = input_h5.data
        self.integralStart = integral_start

    @ti.kernel
    def ComputeIntegralKernel(self, super_samples:int , step_distance: float):
        for x, y, z in self.integralation:
            result = 0.0
            for s in range(super_samples):
                position = GetPositionFromDataIndex(tm.vec3(self.bb_min), tm.vec3(self.cell_width), tm.vec3([x, y, z]), s)
                direction = (position - self.integralStart).normalized()
                distance = (position - self.integralStart).norm()
                intersected_t = IntersectBB(tm.vec3(self.bb_min), tm.vec3(self.bb_max) , tm.vec3(self.integralStart), tm.vec3(direction))
                t0 = intersected_t[0]
                integral_value = self.ComputeIntegralRay(self.integralStart, direction, step_distance, distance, t0)
                result += integral_value
            self.integralation[x, y, z] = result / super_samples
        

    def Compute3dIntegral(self, super_samples: int, step_distance: float):
        ti_field_shape = (self.data.shape[0], self.data.shape[1], self.data.shape[2])
        self.integralation = ti.field(ti.f32, shape=ti_field_shape)
        self.ComputeIntegralKernel(super_samples, step_distance)

    @ti.func
    def ComputeIntegralRay(self, origin, direction, step_distance, maxDistance, t0):
        bb_min = self.bb_min
        bb_max = self.bb_max
        cell_width = self.cell_width
        accumSigmaXStep = 0.0
        max_step = int((maxDistance - t0) / step_distance) + 1
        tmpT = t0
        for step in range(max_step):
            if (tmpT + step_distance < maxDistance):
                position = origin + direction * tmpT
                sigma = InterpolationField(bb_min, bb_max, self.data, position, cell_width)
                accumSigmaXStep += sigma * step_distance
                tmpT += step_distance
            else:
                position = origin + direction * tmpT
                sigma = InterpolationField(bb_min, bb_max, self.data, position, cell_width)
                accumSigmaXStep += sigma * (maxDistance - tmpT)
                tmpT = maxDistance
                break
        return accumSigmaXStep

def main():
    compute_mode = sys.argv[1]
    if compute_mode == 'cpu':
        ti.init(arch=ti.cpu)
    elif compute_mode == 'gpu':
        ti.init(arch=ti.gpu)

    density_file_path = sys.argv[2]
    start_point = sys.argv[3]
    xmlPath = sys.argv[4]
    output_file_path = sys.argv[5]
    
    xml = xml_load(xmlPath)

    cam = xml.get_camera()
    light = xml.get_light()
    step_distance = xml.get_step_distance()
    cell_super_samples = xml.get_cell_super_samples()

    integral_start = tm.vec3([0.0, 0.0, 0.0])
    if start_point == 'camera':
        integral_start = cam.position
    elif start_point == 'light0':
        integral_start = light[0].position
    elif start_point == 'light1':
        integral_start = light[1].position
    elif start_point == 'light2':
        integral_start = light[2].position
    else:
        exit(1)

    integral = Integral3D(density_file_path, integral_start)
    integral.Compute3dIntegral(cell_super_samples, step_distance)
    integral_output = integral.integralation.to_numpy()


    Output3D(integral.bb_min, integral.cell_width, integral_output, output_file_path)

if __name__ == '__main__':
    main()