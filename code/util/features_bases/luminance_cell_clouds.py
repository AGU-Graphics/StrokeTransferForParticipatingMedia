# -----------------------------------------------------------------------------
# Stroke Transfer for Participating Media
# http://cg.it.aoyama.ac.jp/yonghao/sig25/abstsig25.html
#
# File: luminance_cell_clouds.py
# Maintainer: Naoto Shirashima and Yuki Yamaoka
#
# Description:
# Internal brightness calculation for Clouds scenes
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
from black_body import BlackBodyField
from camera import Camera
from light import Light
from data_io import DataIO3D, Output3D
from bounding_box import GetPositionFromDataIndex, IntersectBB
from interpolation_taichi_field import InterpolationField

class ComputeilluminanceSome():
    def __init__(self, whole_density_path: str, density_paths: list[str], cam: Camera, blackbody_field: BlackBodyField):
        input_h5 = DataIO3D.InitLoadFile(whole_density_path)
        self.bb_min = input_h5.bb_min
        self.bb_max = input_h5.bb_max_taichi()
        self.cell_width = input_h5.cell_width
        self.shape = input_h5.get_shape()
        self.density = input_h5.data

        # camera
        self.camera = cam

        self.transmittance_gaussian_filtered = None
        self.transmittance_spline = None

        # blackbody
        self.blackbody_field = blackbody_field
        self.blackbody_bb_min = blackbody_field.bb_min
        self.blackbody_bb_max = blackbody_field.bb_max
        self.blackbody_cell_width = blackbody_field.cell_width
        self.blackbody_resolution = blackbody_field.shape

        self.media_size = len(density_paths)

        h5 = DataIO3D.InitLoadFile(density_paths[0])
        self.media1_bb_min = h5.bb_min
        self.media1_cell_width = h5.cell_width
        self.media1_density = h5.data

        h5 = DataIO3D.InitLoadFile(density_paths[1])
        self.media2_bb_min = h5.bb_min
        self.media2_cell_width = h5.cell_width
        self.media2_density = h5.data


    @ti.func
    def henyey_greenstein_phase_function(self, cos_theta: float, g: float) -> ti.f32:
        return (1.0 - g * g) / (4.0 * tm.pi * tm.pow(1.0 + g * g - 2.0 * g * cos_theta, 1.5))

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

    def ComputeIlluminance(self, lights: list[Light], albedo: list[tm.vec3], phaseFunc_g1: list[float], phaseFunc_g2: list[float], phaseFunc_blend: list[float], step_distance: float , super_samples: int):
        self.Illuminance = ti.field(dtype=ti.f32, shape=(self.shape[0], self.shape[1], self.shape[2], 3))
        cell_width = self.cell_width
        spectral_r = 602.0 * 1.0e-9
        spectral_g = 536.0 * 1.0e-9
        spectral_b = 448.0 * 1.0e-9
        spectral_rgb = tm.vec3([spectral_r, spectral_g, spectral_b])

        lights_nums = len(lights)

        lights_position = ti.Vector.field(3, dtype=ti.f32, shape=lights_nums)
        lights_color = ti.Vector.field(3, dtype=ti.f32, shape=lights_nums)
        lights_distance_attenuation = ti.field(dtype=ti.i16, shape=lights_nums)

        lights_position.from_numpy(np.array([light.position for light in lights]))
        lights_color.from_numpy(np.array([light.color for light in lights]))
        lights_distance_attenuation.from_numpy(np.array([int(light.distance_attenuation) for light in lights]))

        @ti.kernel
        def ComputeIlluminanceMixKernel(step_distance: float, lights_num: int, light_position: ti.template(), light_color: ti.template()):
            bb_min = self.bb_min
            bb_max = self.bb_max
            cell_width = self.cell_width[0]
            media1_bb_min = self.media1_bb_min
            media2_bb_min = self.media2_bb_min

            blackbody_bb_min = self.blackbody_bb_min
            blackbody_bb_max = self.blackbody_bb_max
            blackbody_cell_width = self.blackbody_cell_width

            for I in ti.grouped(self.Illuminance):
                result = tm.vec3([0.0, 0.0, 0.0])
                for s in range(super_samples):
                    position = GetPositionFromDataIndex( bb_min, cell_width*tm.vec3(1,1,1), tm.vec3(I[0], I[1], I[2]), s)
                    for light_index in range(lights_num):
                        direction = (position - light_position[light_index]).normalized()
                        distance = (position - light_position[light_index]).norm()
                        intersected_t = IntersectBB(bb_min, bb_max, tm.vec3(light_position[light_index]), tm.vec3(direction))
                        t0 = intersected_t[0]
                        t1 = intersected_t[1]
                        transmittance = self.ComputeTransmittanceRay(light_position[light_index], direction, step_distance, distance, t0)
                        from_cell_to_eye = (self.camera.position - position).normalized()
                        whole_albedo_g = tm.vec3([0.0, 0.0, 0.0])

                        # media 1
                        phase_function_dot = direction.dot(from_cell_to_eye)
                        phase_function = self.henyey_greenstein_phase_function(phase_function_dot, phaseFunc_g1[0]) * (1.0 - phaseFunc_blend[0]) + self.henyey_greenstein_phase_function(phase_function_dot, phaseFunc_g2[0]) * phaseFunc_blend[0]
                        tmp_bb_max = self.media1_bb_min + self.media1_cell_width * tm.vec3(self.media1_density.shape[2], self.media1_density.shape[1], self.media1_density.shape[0])
                        tmp_bb_cell_size = tm.vec3(self.media1_cell_width[0], self.media1_cell_width[1], self.media1_cell_width[2])
                        whole_albedo_g = whole_albedo_g + albedo[0] * phase_function * InterpolationField(media1_bb_min, tmp_bb_max, self.media1_density, position, tmp_bb_cell_size)

                        # media 2
                        phase_function = self.henyey_greenstein_phase_function(phase_function_dot, phaseFunc_g1[1]) * (1.0 - phaseFunc_blend[1]) + self.henyey_greenstein_phase_function(phase_function_dot, phaseFunc_g2[1]) * phaseFunc_blend[1]
                        tmp_bb_max = self.media2_bb_min + self.media2_cell_width * tm.vec3(self.media2_density.shape[2], self.media2_density.shape[1], self.media2_density.shape[0])
                        tmp_bb_cell_size = tm.vec3(self.media2_cell_width[0], self.media2_cell_width[1], self.media2_cell_width[2])
                        whole_albedo_g = whole_albedo_g + albedo[1] * phase_function * InterpolationField(media2_bb_min, tmp_bb_max, self.media2_density, position, tmp_bb_cell_size)
                        
                        interpolated_density_whole = InterpolationField(bb_min, bb_max, self.density, position, tm.vec3(cell_width,cell_width,cell_width))
                        for i in range(3):
                            if(interpolated_density_whole > 0):
                                whole_albedo_g[i] = whole_albedo_g[i] / interpolated_density_whole

                        distance_attenuation = 1.0
                        if lights_distance_attenuation[light_index] == 1:
                            distance_attenuation = 1.0 / (distance * distance)
                        result = result + transmittance * light_color[light_index] * whole_albedo_g * distance_attenuation
                self.Illuminance[I[0], I[1], I[2], I[3]] = result[I[3]] / super_samples
        ComputeIlluminanceMixKernel(step_distance, lights_nums, lights_position, lights_color)


    def OutputIlluminance(self, output_hdf5_file_r: str, output_hdf5_file_g: str, output_hdf5_file_b: str):
        rgb = self.Illuminance.to_numpy()
        bb_min = np.array(self.bb_min)
        cell_width = np.array(self.cell_width)

        Output3D(bb_min, cell_width, rgb[:, :, :, 0], output_hdf5_file_r)
        Output3D(bb_min, cell_width, rgb[:, :, :, 1], output_hdf5_file_g)
        Output3D(bb_min, cell_width, rgb[:, :, :, 2], output_hdf5_file_b)


def main():
    compute_mode = sys.argv[1]
    if compute_mode == 'cpu':
        ti.init(arch=ti.cpu)
    elif compute_mode == 'gpu':
        ti.init(arch=ti.gpu)
    else:
        raise ValueError("Invalid compute mode. Use 'cpu' or 'gpu'.")

    density_whole_file_path_argv = sys.argv[2]
    media_paths = ['', '']
    media_paths[0] = sys.argv[3]
    media_paths[1] = sys.argv[4]

    xmlPath_argv = sys.argv[5]
    output_illuminance_file_path_R = sys.argv[6]
    output_illuminance_file_path_G = sys.argv[7]
    output_illuminance_file_path_B = sys.argv[8]
    xml = xml_load(xmlPath_argv)

    media1_phase_function_g1 = xml.get_phase_function_HG()
    media1_phase_function_g2 = xml.get_phase_function_HG2()
    media1_phase_function_g_blend = xml.get_phase_function_HG_blend()
    media2_phase_function_g1 = xml.get_phase_function_HG(media_index=2)
    media2_phase_function_g2 = xml.get_phase_function_HG2(media_index=2)
    media2_phase_function_g_blend = xml.get_phase_function_HG_blend(media_index=2)
    temp_min = xml.get_temp_min()
    temp_max = xml.get_temp_max()
    temp_factor = xml.get_temp_factor()
    blackbody_factor = xml.get_blackbody_factor()
    # blackbody_file_path = xml.get_temperature_file_path()


    cell_super_samples = xml.get_cell_super_samples()
    screen_super_samples = xml.get_screen_super_samples()
    step_distance = xml.get_step_distance()

    cam = xml.get_camera()
    light = xml.get_light()

    media1_albedo = xml.get_albedo()
    media2_albedo = xml.get_albedo(media_index=2)

    # blackbody = BlackBodyField(temp_min, temp_max, temp_factor, blackbody_factor, blackbody_file_path)
    blackbody = BlackBodyField(temp_min, temp_max, temp_factor, blackbody_factor, None)
    illuminance_some = ComputeilluminanceSome(whole_density_path=density_whole_file_path_argv, density_paths=media_paths , cam=cam, blackbody_field=blackbody)
    illuminance_some.ComputeIlluminance(light, [media1_albedo, media2_albedo], [media1_phase_function_g1, media2_phase_function_g1], [media1_phase_function_g2, media2_phase_function_g2], [media1_phase_function_g_blend, media2_phase_function_g_blend],step_distance=step_distance , super_samples=cell_super_samples)
    illuminance_some.OutputIlluminance(output_illuminance_file_path_R, output_illuminance_file_path_G, output_illuminance_file_path_B)

if __name__ == '__main__':
    main()
