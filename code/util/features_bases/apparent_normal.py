# -----------------------------------------------------------------------------
# Stroke Transfer for Participating Media
# http://cg.it.aoyama.ac.jp/yonghao/sig25/abstsig25.html
#
# File: apparent_normal.py
# Maintainer: Naoto Shirashima and Yuki Yamaoka
#
# Description:
# Transform the transmittance on the grid into a normal using Cubic-B-Spline
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
import sys
import taichi as ti
import taichi.math as tm
from xml_load import xml_load
from depth_value import DepthValue
from camera import Camera
from data_io import DataIO3D, Output2D
from integral_conversion import ComputeScreenIntegral
from bounding_box import IntersectBB,  ClampPositionByBB
from interpolation_taichi_field import CubicBSpline3D, InterpolationField


class ComputeApparentNormal:
    def __init__(self, transmittance_data_io: ti.template(), fpd_data_io: ti.template(), depth: DepthValue, camera: Camera, step_distance: float):
        self.fpd_field = fpd_data_io.data

        self.camera = camera
        self.bb_min = tm.vec3(fpd_data_io.bb_min)
        self.bb_max = fpd_data_io.bb_max_taichi()
        self.cell_width = transmittance_data_io.cell_width
        self.shape = transmittance_data_io.get_shape()
        self.transmittance_spline = CubicBSpline3D.InitByNumpy(transmittance_data_io.data_np, self.bb_min, self.cell_width[0])
        self.depth = depth
        self.step_distance = step_distance


    def ComputeScreenSpaceApparentNormal(self, super_samples: int):
        result = ComputeScreenIntegral(3, self.step_distance, self.camera, self.depth, super_samples, self.ComputeIntegralApparentNormal).to_numpy()
        return result

    @ti.func
    def ComputeIntegralApparentNormal(self, ray_org: tm.vec3, ray_dir: tm.vec3, step_distance: float, depth: float):
        apparent_normal_epsilon = 1e-5
        bb_min = self.bb_min
        bb_max = self.bb_max
        grad_norm = tm.vec3([0.0, 0.0, 0.0])

        # print('cell width', self.cell_width[0], self.cell_width[1], self.cell_width[2])
        inner_index = 0
        intersect_bb_min = self.bb_min + self.cell_width * tm.vec3([inner_index, inner_index, inner_index])
        intersect_bb_max = self.bb_min + self.cell_width * tm.vec3([ self.shape[0] - inner_index -1, self.shape[1] - inner_index - 1, self.shape[2] - inner_index - 1])
        intersect_t = IntersectBB(intersect_bb_min, intersect_bb_max, ray_org, ray_dir)
        t0 = intersect_t[0]
        t1 = intersect_t[1]
        if not ti.math.isinf(depth) and not ti.math.isnan(depth) and depth < t1:
            t1 = depth
        result = tm.vec3([0.0, 0.0, 0.0])
        max_step = int((t1 - t0) / step_distance) + 1
        tmp_t = t0
        for now_step in range(max_step):
            if tmp_t + step_distance < t1:
                position = ray_org + ray_dir * tmp_t
                position = ClampPositionByBB(position, bb_min, bb_max)
                grad = self.transmittance_spline.Gradient(position)
                if tm.length(grad) < apparent_normal_epsilon:
                    grad_norm = [0.0, 0.0, 0.0]
                else:
                    grad_norm = grad.normalized()
                pdf = InterpolationField(bb_min, bb_max, self.fpd_field, position, tm.vec3(self.cell_width))
                result = result + pdf * grad_norm * step_distance
                tmp_t += step_distance
            else:
                remain_distance = t1 - tmp_t
                position = ray_org + ray_dir * tmp_t
                position = ClampPositionByBB(position, bb_min, bb_max)
                grad = self.transmittance_spline.Gradient(position)
                if tm.length(grad) < apparent_normal_epsilon:
                    grad_norm = [0.0, 0.0, 0.0]
                else:
                    grad_norm = grad.normalized()
                pdf = InterpolationField(bb_min, bb_max, self.fpd_field, position, tm.vec3(self.cell_width))
                result = result + pdf * grad_norm * remain_distance
                tmp_t = t1
        return result

def main():
    compute_mode = sys.argv[1]
    if compute_mode == 'cpu':
        ti.init(arch=ti.cpu)
    elif compute_mode == 'gpu':
        ti.init(arch=ti.gpu)

    transmittance_file_path = sys.argv[2]
    fpd_file_path = sys.argv[3]
    depth_file_path = sys.argv[4]
    xmlPath = sys.argv[5]
    output_apparent_normal_file_path = sys.argv[6]

    xml = xml_load(xmlPath)

    cam = xml.get_camera()
    step_distance = xml.get_step_distance()
    screen_super_samples = xml.get_screen_super_samples()
    camera_pixels = cam.pixels

    transmittance_cell = DataIO3D.InitLoadFile( transmittance_file_path, taichi=False )
    fpd = DataIO3D.InitLoadFile( fpd_file_path )
    depth = DepthValue(depth_file_path)
    compute_apparent_normal = ComputeApparentNormal(transmittance_cell, fpd, depth, cam, step_distance)
    apparent_normal_screen = compute_apparent_normal.ComputeScreenSpaceApparentNormal(super_samples=screen_super_samples)

    Output2D(apparent_normal_screen, output_apparent_normal_file_path)

if __name__ == '__main__':
    main()
