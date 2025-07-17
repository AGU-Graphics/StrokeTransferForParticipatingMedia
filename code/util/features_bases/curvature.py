# -----------------------------------------------------------------------------
# Stroke Transfer for Participating Media
# http://cg.it.aoyama.ac.jp/yonghao/sig25/abstsig25.html
#
# File: curvature.py
# Maintainer: Naoto Shirashima and Yuki Yamaoka
#
# Description:
# curvature calculation
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
import numpy as np
from xml_load import xml_load
from data_io import Output2D, DataIO3D
from depth_value import DepthValue
from camera import Camera
from integral_conversion import ComputeScreenIntegral
from bounding_box import IntersectBB, ClampPositionByBB
from interpolation_taichi_field import CubicBSpline3D, InterpolationField

class ComputeCurvature:
    def __init__(self, transmittance_cell_path:str , fpd_path: str, depth: DepthValue, camera: Camera, step_distance: float):
        self.fpd_field = DataIO3D.InitLoadFile(fpd_path).data
        self.camera = camera
        transmittance_data_io = DataIO3D.InitLoadFile(transmittance_cell_path)
        self.bb_min = tm.vec3(transmittance_data_io.bb_min)
        self.bb_max = tm.vec3(transmittance_data_io.bb_max_taichi())
        self.cell_width = transmittance_data_io.cell_width
        self.transmittance_gaussian_filtered = CubicBSpline3D.InitByNumpy(transmittance_data_io.data_np, self.bb_min, self.cell_width[0])
        self.depth = depth
        self.step_distance = step_distance

    def ComputeScreenSpaceGaussianCurvature(self, super_samples: int):
        result = ComputeScreenIntegral(1, self.step_distance, self.camera, self.depth, super_samples, self.ComputeIntegralGaussianCurvature).to_numpy()
        return result

    @ti.func
    def ComputeIntegralGaussianCurvature(self, ray_org: tm.vec3, ray_dir: tm.vec3, step_distance: float, depth: float):
        gaussian_curvature_epsilon = 1e-5
        bb_min = self.bb_min
        bb_max = self.bb_max
        # print("bb_min", bb_min)
        # print("bb_max", bb_max)
        intersect_t = IntersectBB(tm.vec3(self.bb_min), tm.vec3(self.bb_max), ray_org, ray_dir)
        t0 = intersect_t[0]
        t1 = intersect_t[1]
        if not ti.math.isinf(depth) and not ti.math.isnan(depth) and depth < t1:
            t1 = depth
        result = 0.0
        max_step = int((t1 - t0) / step_distance) + 1
        tmp_t = t0
        for now_step in range(max_step):
            if tmp_t + step_distance < t1:
                position = ray_org + ray_dir * tmp_t
                position = ClampPositionByBB(position, bb_min, bb_max)
                grad = self.transmittance_gaussian_filtered.Gradient(position)
                hesse = self.transmittance_gaussian_filtered.Gradient2(position)
                fpd = InterpolationField(bb_min, bb_max, self.fpd_field, position, tm.vec3(self.cell_width))
                result += fpd * self.ComputeGaussianCurvature(grad, hesse,
                                                                                    gaussian_curvature_epsilon) * step_distance
                tmp_t += step_distance
            else:
                remain_distance = t1 - tmp_t
                position = ray_org + ray_dir * tmp_t
                position = ClampPositionByBB(position, bb_min, bb_max)
                grad = self.transmittance_gaussian_filtered.Gradient(position)
                hesse = self.transmittance_gaussian_filtered.Gradient2(position)
                fpd = InterpolationField(bb_min, bb_max, self.fpd_field, position, tm.vec3(self.cell_width))
                result += fpd * self.ComputeGaussianCurvature(grad, hesse,
                                                                                    gaussian_curvature_epsilon) * remain_distance
                tmp_t = t1
        return result

    @staticmethod
    @ti.func
    def ComputeGaussianCurvature(grad: tm.vec3, hesse: tm.mat3, epsilon: ti.f32) -> ti.f32:
        result = ti.f32(0.0)
        if grad.norm() > epsilon:
            HSF = tm.mat3(
                [
                    hesse[1, 1] * hesse[2, 2] - hesse[1, 2] * hesse[2, 1],
                    hesse[1, 2] * hesse[2, 0] - hesse[1, 0] * hesse[2, 2],
                    hesse[1, 0] * hesse[2, 1] - hesse[1, 1] * hesse[2, 0]
                ],
                [
                    hesse[2, 1] * hesse[0, 2] - hesse[2, 2] * hesse[0, 1],
                    hesse[2, 2] * hesse[0, 0] - hesse[2, 0] * hesse[0, 2],
                    hesse[2, 0] * hesse[0, 1] - hesse[2, 1] * hesse[0, 0]
                ],
                [
                    hesse[0, 1] * hesse[1, 2] - hesse[0, 2] * hesse[1, 1],
                    hesse[0, 2] * hesse[1, 0] - hesse[0, 0] * hesse[1, 2],
                    hesse[0, 0] * hesse[1, 1] - hesse[0, 1] * hesse[1, 0]
                ]
            )
            norm_ = grad.norm() ** 4
            result = (grad.dot(HSF @ grad)) / (norm_)
        return result

    def ComputeScreenSpaceMeanCurvature(self, super_samples: int):
        result = ComputeScreenIntegral(1, self.step_distance, self.camera, self.depth, super_samples, self.ComputeIntegralMeanCurvature).to_numpy()
        return result

    @ti.func
    def ComputeIntegralMeanCurvature(self, ray_org: tm.vec3, ray_dir: tm.vec3, step_distance: float, depth: float):
        gaussian_curvature_epsilon = 1e-5
        bb_min = self.bb_min
        bb_max = self.bb_max
        intersect_t = IntersectBB(tm.vec3(self.bb_min), tm.vec3(self.bb_max), ray_org, ray_dir)
        t0 = intersect_t[0]
        t1 = intersect_t[1]
        if not ti.math.isinf(depth) and not ti.math.isnan(depth) and depth < t1:
            t1 = depth
        result = 0.0
        max_step = int((t1 - t0) / step_distance) + 1
        tmp_t = t0
        for now_step in range(max_step):
            if tmp_t + step_distance < t1:
                position = ray_org + ray_dir * tmp_t
                position = ClampPositionByBB(position, bb_min, bb_max)
                grad = self.transmittance_gaussian_filtered.Gradient(position)
                hesse = self.transmittance_gaussian_filtered.Gradient2(position)
                fpd = InterpolationField(bb_min, bb_max, self.fpd_field, position, tm.vec3(self.cell_width))
                result += fpd * self.ComputeMeanCurvature(grad, hesse,
                                                                             gaussian_curvature_epsilon) * step_distance
                tmp_t += step_distance
            else:
                remain_distance = t1 - tmp_t
                position = ray_org + ray_dir * tmp_t
                position = ClampPositionByBB(position, bb_min, bb_max)
                grad = self.transmittance_gaussian_filtered.Gradient(position)
                hesse = self.transmittance_gaussian_filtered.Gradient2(position)
                fpd = InterpolationField(bb_min, bb_max, self.fpd_field, position, tm.vec3(self.cell_width))
                result += fpd * self.ComputeMeanCurvature(grad, hesse,
                                                                             gaussian_curvature_epsilon) * remain_distance
                tmp_t = t1
        return result

    @staticmethod
    @ti.func
    def ComputeMeanCurvature(grad: tm.vec3, hesse: tm.mat3, epsilon: ti.f32) -> ti.f32:
        result = ti.f32(0.0)
        if grad.norm() > epsilon:
            norm3 = grad.norm() ** 3
            norm2 = grad.norm() ** 2
            result = -(grad.dot(hesse @ grad) - norm2 * hesse.trace()) / (2.0 * norm3)
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
    output_gaussian_curvature_file_path = sys.argv[6]
    output_mean_curvature_file_path = sys.argv[7]

    xml = xml_load(xmlPath)
    cam = xml.get_camera()
    step_distance = xml.get_step_distance()
    screen_super_samples = xml.get_screen_super_samples()
    depth = DepthValue(depth_file_path)
    computeCurvature = ComputeCurvature(transmittance_file_path, fpd_file_path, depth, cam, step_distance)

    gaussian_curvature_screen = computeCurvature.ComputeScreenSpaceGaussianCurvature(super_samples=screen_super_samples)
    mean_curvature_screen = computeCurvature.ComputeScreenSpaceMeanCurvature(super_samples=screen_super_samples)

    Output2D(gaussian_curvature_screen, output_gaussian_curvature_file_path)
    Output2D(mean_curvature_screen, output_mean_curvature_file_path)


if __name__ == '__main__':
    main()
