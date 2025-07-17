# -----------------------------------------------------------------------------
# Stroke Transfer for Participating Media
# http://cg.it.aoyama.ac.jp/yonghao/sig25/abstsig25.html
#
# File: luminance_cell_mix_media.py
# Maintainer: Naoto Shirashima and Yuki Yamaoka
#
# Description:
# Inner brightness calculation for CollidingSmokes
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
import argparse
import taichi as ti
import taichi.math as tm
import numpy as np
from data_io import DataIO3D, Output3D
from xml_load import xml_load
from camera import Camera
from light import Light
from black_body import BlackBodyField
from bounding_box import GetPositionFromDataIndex, IntersectBB
from interpolation_taichi_field import InterpolationField

@ti.data_oriented
class ComputeilluminanceMix():
    def __init__(self, hdf5_path: str, cam: Camera, phase_function_g: float, blackbody_field: BlackBodyField):
        input_h5 = DataIO3D.InitLoadFile(hdf5_path)
        self.bb_min = input_h5.bb_min
        self.bb_max = input_h5.bb_max_taichi()
        self.cell_width = input_h5.cell_width
        self.resolution = input_h5.get_shape()
        self.density = input_h5.data
        self.phase_function_g = phase_function_g
        self.camera = cam
        self.transmittance_gaussian_filtered = None
        self.transmittance_spline = None
        self.blackbody_field = blackbody_field
        self.blackbody_bb_min = blackbody_field.bb_min
        self.blackbody_bb_max = blackbody_field.bb_max
        self.blackbody_cell_width = blackbody_field.cell_width
        self.blackbody_resolution = blackbody_field.shape


#     @ti.func
#     def henyey_greenstein_phase_function(self, w_in: tm.vec3, w_out: tm.vec3) -> ti.f32:
#         g = self.phase_function_g
#         cos_theta = w_in.dot(w_out)
#         return (1.0 - g * g) / (4.0 * tm.pi * tm.pow(1.0 + g * g - 2.0 * g * cos_theta, 1.5))

    @ti.func
    def ComputeTransmittanceRay(self, origin, direction, step_distance, maxDistance, t0):
        bb_min = self.bb_min
        bb_max = self.bb_max
        cell_width = self.cell_width
        accumSigmaXStep = 0.0
        max_step = int((maxDistance - t0) / step_distance) + 1
        tmpT = t0
        for step in range(max_step):
            if (tmpT + step_distance < maxDistance):
                position = origin + direction * tmpT
                sigma = InterpolationField(bb_min, bb_max, self.density, position, cell_width)
                accumSigmaXStep += sigma * step_distance
                tmpT += step_distance
            else:
                position = origin + direction * tmpT
                sigma = InterpolationField(bb_min, bb_max, self.density, position, cell_width)
                accumSigmaXStep += sigma * step_distance
                tmpT += maxDistance
                break
        return tm.exp(-accumSigmaXStep)

    @ti.kernel
    def ComputeIlluminanceMixKernel(
        self,
        super_samples: int,
        step_distance: float,
        lights_num: int,
        phaseFunc_g: float,
        albedo: tm.vec3,
        albedo2: tm.vec3,
        light_position: ti.template(),
        light_color: ti.template(),
        lights_distance_attenuation: ti.template(),
    ):
        bb_min = self.bb_min
        bb_max = self.bb_max
        cell_width = self.cell_width

        blackbody_bb_min = self.blackbody_bb_min
        blackbody_bb_max = self.blackbody_bb_max
        blackbody_cell_width = self.blackbody_cell_width

        for I in ti.grouped(self.Illuminance):
            result = tm.vec3([0.0, 0.0, 0.0])
            for s in range(super_samples):
                position = GetPositionFromDataIndex(bb_min, cell_width, tm.vec3([I[0], I[1], I[2]]), s)
                for light_index in range(lights_num):
                    direction = (position - light_position[light_index]).normalized()
                    distance = (position - light_position[light_index]).norm()
                    intersected_t = IntersectBB(bb_min, bb_max, tm.vec3(light_position[light_index]), tm.vec3(direction))
                    t0 = intersected_t[0]
                    t1 = intersected_t[1]
                    transmittance = self.ComputeTransmittanceRay(light_position[light_index], direction, step_distance, distance, t0)
                    from_cell_to_eye = (self.camera.position - position).normalized()
                    dot = direction.dot(from_cell_to_eye)
                    phaseFunc = 0.5 * (1.0 - phaseFunc_g * phaseFunc_g) / (
                        tm.pow(1.0 + phaseFunc_g * phaseFunc_g - 2.0 * phaseFunc_g * dot, 3.0 / 2))
                    albedo_blend =min(1.0, InterpolationField(blackbody_bb_min, blackbody_bb_max, self.blackbody_field.data, position, blackbody_cell_width) * 1000.0)
                    total_albedo = albedo_blend * albedo + (1.0 - albedo_blend) * albedo2
                    distance_attenuation = 1.0
                    if lights_distance_attenuation[light_index] == 1:
                        distance_attenuation = 1.0 / (distance * distance)
                    scatter = transmittance * light_color[light_index] * phaseFunc * total_albedo * distance_attenuation
                    result = result + scatter

            self.Illuminance[I[0], I[1], I[2], I[3]] = result[I[3]] / super_samples

    def ComputeIlluminance(self, lights: list[Light], albedo: tm.vec3, albedo2: tm.vec3, phaseFunc_g: float, step_distance: float , super_samples: int):
        self.Illuminance = ti.field(dtype=ti.f32, shape=(self.resolution[0], self.resolution[1], self.resolution[2], 3))
        phaseFunc_g = phaseFunc_g
        albedo = albedo
        spectral_r = 602.0 * 1.0e-9
        spectral_g = 536.0 * 1.0e-9
        spectral_b = 448.0 * 1.0e-9
        # spectral_rgb = tm.vec3([spectral_r, spectral_g, spectral_b])

        lights_nums = len(lights)

        lights_position = ti.Vector.field(3, dtype=ti.f32, shape=lights_nums)
        lights_color = ti.Vector.field(3, dtype=ti.f32, shape=lights_nums)
        lights_distance_attenuation = ti.field(dtype=ti.i16, shape=lights_nums)

        lights_position.from_numpy(np.array([light.position for light in lights]))
        lights_color.from_numpy(np.array([light.color for light in lights]))
        lights_distance_attenuation.from_numpy(np.array([int(light.distance_attenuation) for light in lights]))


        self.ComputeIlluminanceMixKernel(
            super_samples=super_samples,
            step_distance=step_distance,
            lights_num=lights_nums,
            light_position=lights_position,
            light_color=lights_color,
            lights_distance_attenuation=lights_distance_attenuation,
            phaseFunc_g=phaseFunc_g,
            albedo=albedo,
            albedo2=albedo2,
        )

    def OutputIlluminance(self, output_hdf5_file_r: str, output_hdf5_file_g: str, output_hdf5_file_b: str):
        rgb = self.Illuminance.to_numpy()
        bb_min = np.array(self.bb_min)
        bb_max = np.array(self.bb_max)
        cell_width = np.array(self.cell_width)
        Output3D(bb_min, cell_width, rgb[:, :, :, 0], output_hdf5_file_r)
        Output3D(bb_min, cell_width, rgb[:, :, :, 1], output_hdf5_file_g)
        Output3D(bb_min, cell_width, rgb[:, :, :, 2], output_hdf5_file_b)


def main():
    args = argparse.ArgumentParser()
    args.add_argument('-compute_mode', type=str, required=True, help='cpu or gpu')
    args.add_argument('-extinction', type=str, required=True, help='extinction file path')
    args.add_argument('-xml', type=str, required=True, help='xml file path')
    args.add_argument('-out_r', type=str, required=True, help='output illuminance file path R')
    args.add_argument('-out_g', type=str, required=True, help='output illuminance file path G')
    args.add_argument('-out_b', type=str, required=True, help='output illuminance file path B')
    args.add_argument('-temperature', type=str, help='temperature file path', default=None)
    args = args.parse_args()

    density_file_path = args.extinction
    temperature_file_path = args.temperature
    xmlPath = args.xml
    output_illuminance_file_path_R = args.out_r
    output_illuminance_file_path_G = args.out_g
    output_illuminance_file_path_B = args.out_b
    xml = xml_load(xmlPath)

    phase_function_HG = xml.get_phase_function_HG()
    cam = xml.get_camera()
    light = xml.get_light()
    albedo = xml.get_albedo()
    albedo2 = xml.get_albedo2()
    temp_min = xml.get_temp_min()
    temp_max = xml.get_temp_max()
    temp_factor = xml.get_temp_factor()
    blackbody_factor = xml.get_blackbody_factor()
    cell_super_samples = xml.get_cell_super_samples()
    screen_super_samples = xml.get_screen_super_samples()
    step_distance = xml.get_step_distance()

    if args.compute_mode == 'cpu':
        ti.init(arch=ti.cpu)
    elif args.compute_mode == 'gpu':
        ti.init(arch=ti.gpu)
    else:
        raise ValueError("Invalid compute mode. Use 'cpu' or 'gpu'.")

    blackbody = BlackBodyField(temp_min, temp_max, temp_factor, blackbody_factor, temperature_file_path)
    illuminance_mix = ComputeilluminanceMix(hdf5_path=density_file_path, cam=cam, phase_function_g=phase_function_HG, blackbody_field=blackbody)
    illuminance_mix.ComputeIlluminance(light, albedo, albedo2, phase_function_HG, step_distance=step_distance , super_samples=cell_super_samples)
    illuminance_mix.OutputIlluminance(output_illuminance_file_path_R, output_illuminance_file_path_G, output_illuminance_file_path_B)

if __name__ == '__main__':
    main()
