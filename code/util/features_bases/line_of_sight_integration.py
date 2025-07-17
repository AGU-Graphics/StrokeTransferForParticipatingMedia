# -----------------------------------------------------------------------------
# Stroke Transfer for Participating Media
# http://cg.it.aoyama.ac.jp/yonghao/sig25/abstsig25.html
#
# File: line_of_sight_integration.py
# Maintainer: Naoto Shirashima and Yuki Yamaoka
#
# Description:
# Perform line-of-sight integration on volume data
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
import numpy as np
import argparse
from xml_load import xml_load
from depth_value import DepthValue
from data_io import DataIO3D, Output2D
from camera import Camera
from integral_conversion import ComputeScreenIntegral
from bounding_box import IntersectBB
from interpolation_taichi_field import InterpolationField

class LineOfSightIntegration:
    def __init__(self, density_file_path, integral_start, depth: DepthValue, camera: Camera, step_distance: float):
        input_h5 = DataIO3D.InitLoadFile(density_file_path)

        self.bb_min = input_h5.bb_min
        self.bb_max = input_h5.bb_max_taichi()
        self.cell_length = input_h5.cell_width
        self.shape = input_h5.get_shape()
        self.data = input_h5.data
        self.integralStart = integral_start
        self.depth = depth
        self.camera = camera
        self.step_distance = step_distance

    def Compute3d_to_2d_integral(self, super_samples: int):
        result = ComputeScreenIntegral(1, self.step_distance, self.camera, self.depth, super_samples, self.ComputeIntegral).to_numpy()
        return result

    @ti.func
    def ComputeIntegral(self, ray_org: tm.vec3, ray_dir: tm.vec3, step_distance: float, depth: float):
        bb_min = self.bb_min
        bb_max = self.bb_max
        cell_length = self.cell_length
        intersect_t = IntersectBB(bb_min, bb_max, ray_org, ray_dir)
        t0 = intersect_t[0]
        t1 = intersect_t[1]
        accumSigmaXStep = 0.0

        if not ti.math.isinf(depth) and not ti.math.isnan(depth) and depth < t1:
            t1 = depth

        max_step = int((t1 - t0) / step_distance) + 1
        tmp_t = t0

        for now_step in range(max_step):
            if tmp_t + step_distance < t1:
                position = ray_org + ray_dir * tmp_t
                sigma = InterpolationField(bb_min, bb_max, self.data, position, cell_length)
                accumSigmaXStep = accumSigmaXStep + sigma * step_distance
                tmp_t = tmp_t + step_distance

            else:
                remain_distance = t1 - tmp_t
                position = ray_org + ray_dir * tmp_t
                sigma = InterpolationField(bb_min, bb_max, self.data, position, cell_length)
                accumSigmaXStep = accumSigmaXStep + sigma * remain_distance
                tmp_t = t1
                break
        return accumSigmaXStep


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('compute_mode', type=str, help='cpu or gpu')
    parser.add_argument('density_file_path', type=str, help='density file path')
    parser.add_argument('depth_test_file_path', type=str, help='depth test file path')
    parser.add_argument('start_point', type=str, help='start point')
    parser.add_argument('xml_path', type=str, help='xml path')
    parser.add_argument('output_file_path', type=str, help='output file path')
    parser.add_argument('--extra_resolution', type=float, default=0, help='extra resolution')
    parser.add_argument('--extra_depth_test', type=str, default='')
    parser.add_argument('--output_extra_resolution', type=str)
    args = parser.parse_args()

    density_file_path = args.density_file_path
    depth_test_file_path = args.depth_test_file_path
    start_point = args.start_point
    xmlPath = args.xml_path
    output_file_path = args.output_file_path
    extra_resolution = args.extra_resolution
    extra_depth_test = args.extra_depth_test
    extra_resolution_output = args.output_extra_resolution

    if args.compute_mode == 'cpu':
        ti.init(arch=ti.cpu)
    elif args.compute_mode == 'gpu':
        ti.init(arch=ti.gpu)
    else:
        print("Invalid compute mode. Use 'cpu' or 'gpu'.")
        exit(-1)

    xml = xml_load(xmlPath)

    cam = xml.get_camera()
    light = xml.get_light()
    step_distance = xml.get_step_distance()

    screen_super_samples = xml.get_screen_super_samples()

    integral_start = tm.vec3([0.0, 0.0, 0.0])
    if start_point == 'camera':
        integral_start = cam.position
    elif start_point == 'light1':
        integral_start = light[0].position
    elif start_point == 'light2':
        integral_start = light[1].position
    else:
        exit(-1)

    depth_value = DepthValue(depth_test_file_path)
    integral = LineOfSightIntegration(density_file_path, integral_start, depth_value, cam, step_distance)
    result = integral.Compute3d_to_2d_integral(screen_super_samples)
    Output2D(result, output_file_path)

    if extra_resolution > 0:
        pixels = np.array(cam.pixels * extra_resolution).astype(int)
        print('extra resolution pixels', pixels)
        fovx = 2 * np.arctan( extra_resolution * cam.fovxHalfLength)
        fovy = 2 * np.arctan( extra_resolution * cam.fovyHalfLength)
        cam_extra = Camera(cam.position, cam.direction, pixels, cam.film, [fovx, fovy])
        integral = LineOfSightIntegration(density_file_path, integral_start, DepthValue(extra_depth_test), cam_extra, step_distance)
        result = integral.Compute3d_to_2d_integral(screen_super_samples)
        Output2D(result, extra_resolution_output)
        print('extra shape', result.shape)


if __name__ == '__main__':
    main()
