# -----------------------------------------------------------------------------
# Stroke Transfer for Participating Media
# http://cg.it.aoyama.ac.jp/yonghao/sig25/abstsig25.html
#
# File: transmittance_surface_screen.py
# Maintainer: Naoto Shirashima and Yuki Yamaoka
#
# Description:
# The transmittance of light through the surface seen by the screen
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
import numpy as np
import sys
import taichi as ti
import taichi.math as tm
from xml_load import xml_load
from depth_value import DepthValue
from data_io import DataIO3D, DataIO2D, Output2D
from camera import Camera
from sample_points_generator import vu_point_to_uv_point_with_haltonseq
from interpolation_taichi_field import InterpolationField

@ti.data_oriented
class ComputeTransmittanceSurface:
    def __init__(self, extinction_path, xml:xml_load ,depth_value: DepthValue, pixel, rgb:bool):
        bb = DataIO3D.InitLoadFile(extinction_path)
        self.extinction = bb.data
        self.bb_min = bb.bb_min
        self.bb_max = bb.bb_max_taichi()
        self.cell_width = bb.cell_width
        self.depth_test = depth_value
        self.rgb = rgb
        if rgb == False:
            self.transmittance_surface = ti.field(dtype=ti.f32, shape=(pixel[1], pixel[0]))
        else:
            self.transmittance_surface = ti.field(dtype=tm.vec3, shape=(pixel[1], pixel[0]))
        self.xml = xml
        self.lights = xml.get_light()

    @ti.kernel
    def ComputeTransmittanceSurfaceKernel(
        self,
        super_samples: int,
        step_distance: float,
        camera: ti.template(),
        num_lights: int,
        light_positions: ti.template(),
        light_colors: ti.template()
    ):
        bb_min = tm.vec3(self.bb_min)
        bb_max = tm.vec3(self.bb_max)
        screen_bb_min = tm.vec2(DataIO2D.screen_bb_min_0, DataIO2D.screen_bb_min_1)
        screen_cell_width = DataIO2D.screen_cell_width
        bb_center = (bb_max + bb_min) * 0.5
        bb_size = (bb_max - bb_min).norm()
        for I in ti.grouped(self.transmittance_surface):
            result = tm.vec3(0,0,0)
            vu = tm.vec2([I[0], I[1]])
            for ss in range(super_samples):
                org, dir = camera.OrgDirFromVU(vu, ss)
                uv = vu_point_to_uv_point_with_haltonseq(vu, ss)
                uv01 = uv / tm.vec2([camera.pixels[0], camera.pixels[1]])
                sample_depth = self.depth_test.GetDepth01(vu_01=tm.vec2(uv01[1], uv01[0]))
                path_through = False
                if tm.isnan(sample_depth) or tm.isinf(sample_depth):
                    path_through = True
                if path_through:
                    result += tm.vec3(1,1,1)
                else:
                    cell_position = org + dir * sample_depth
                    for light_index in range(num_lights):
                        dir = (light_positions[light_index] - cell_position).normalized()
                        max_distance = (light_positions[light_index] - cell_position).norm()
                        attenuation = 1.0
                        result += tm.exp(-self.ComputeIntegralRay(cell_position, dir, step_distance, max_distance, 0.0)) * light_colors[light_index] * attenuation
            if self.rgb:
                self.transmittance_surface[I] = result / super_samples
            else:
                self.transmittance_surface[I] = result[0] / super_samples

    def ComputeTransmittanceSurface(self, step_distance , super_samples):
        bb_min_np = self.bb_min
        bb_max_np = self.bb_max
        num_lights = len(self.lights)
        camera = self.xml.get_camera()
        lights_nums = len(self.lights)
        lights = self.xml.get_light()
        lights_position = ti.Vector.field(3, dtype=ti.f32, shape=lights_nums)
        lights_color = ti.Vector.field(3, dtype=ti.f32, shape=lights_nums)

        lights_position.from_numpy(np.array([light.position for light in lights]))
        lights_color.from_numpy(np.array([light.color for light in lights]))

        self.ComputeTransmittanceSurfaceKernel(
            super_samples=super_samples,
            step_distance=step_distance,
            camera=camera,
            num_lights=lights_nums,
            light_positions=lights_position,
            light_colors=lights_color
        )


    @ti.func
    def ComputeIntegralRay(self, origin, direction, step_distance, maxDistance, t0):
        bb_min = self.bb_min
        bb_max = self.bb_max
        cell_length = self.cell_width
        accumSigmaXStep = 0.0
        max_step = int((maxDistance - t0) / step_distance) + 1
        tmpT = t0
        for step in range(max_step):
            if (tmpT + step_distance < maxDistance):
                position = origin + direction * tmpT
                sigma = InterpolationField(bb_min, bb_max, self.extinction, position, cell_length, clamp_mode=0)
                accumSigmaXStep += sigma * step_distance
                tmpT += step_distance
            else:
                position = origin + direction * tmpT
                sigma = InterpolationField(bb_min, bb_max, self.extinction, position, cell_length, clamp_mode=0)
                accumSigmaXStep += sigma * (maxDistance - tmpT)
                tmpT = maxDistance
                break
        return accumSigmaXStep

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--compute_mode', type=str, default='gpu', choices=['cpu', 'gpu'], help='Compute mode: cpu or gpu')
    parser.add_argument('--extinction_path', type=str, required=True)
    parser.add_argument('--depth_value', type=str, default=None)
    parser.add_argument('--xml', type=str, required=True)
    parser.add_argument('--output', type=str, required=True)
    parser.add_argument('--rgb', action='store_true', help='If it is true, it computes surface reflection "RGB" by xml lights. If it is false, it computes surface reflection only R')
    parser.add_argument('--plot', action='store_true', help='plot by matplot')
    arg = parser.parse_args()

    compute_mode = arg.compute_mode    
    extinction_path = arg.extinction_path
    depth_value = arg.depth_value
    xml = arg.xml
    output = arg.output
    compute_mode = arg.compute_mode
    rgb = arg.rgb
    plot = arg.plot

    if compute_mode == 'cpu':
        ti.init(arch=ti.cpu)
    elif compute_mode == 'gpu':
        ti.init(arch=ti.gpu)

    xml = xml_load(xml)
    camera = xml.get_camera()
    super_samples = xml.get_screen_super_samples()
    step_distance = xml.get_step_distance()


    depth_value = DepthValue(depth_value)
    c = ComputeTransmittanceSurface(extinction_path, xml, depth_value, pixel=camera.pixels, rgb=rgb)
    c.ComputeTransmittanceSurface(step_distance, super_samples)

    result = c.transmittance_surface.to_numpy()
    if rgb:
        Output2D(result[:,:,0], f'{output}_0')
        Output2D(result[:,:,1], f'{output}_1')
        Output2D(result[:,:,2], f'{output}_2')
    else:
        Output2D(result, output)

if __name__ == '__main__':
    main()