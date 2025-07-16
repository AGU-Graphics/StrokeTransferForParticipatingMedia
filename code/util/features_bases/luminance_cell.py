# -----------------------------------------------------------------------------
# Stroke Transfer for Participating Media
# http://cg.it.aoyama.ac.jp/yonghao/sig25/abstsig25.html
#
# File: luminance_cell.py
# Maintainer: Naoto Shirashima
#
# Description:
# Calculate the internal brightness on the grid
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
import argparse
import xml.etree.ElementTree as ET
import numpy as np
from xml_load import xml_load
from black_body import BlackBodyField
from data_io import DataIO3D, Output3D
from light import Light
from bounding_box import GetPositionFromDataIndex, IntersectBB
from interpolation_taichi_field import InterpolationField

@ti.func
def ComputeTransmittanceRay(
    extinction_field,
    bb_min,
    bb_max,
    bb_cell_width,
    ray_origin,
    ray_direction,
    ray_t0,
    step_distance,
    max_distance,
):
    accumSigmaXStep = 0.0
    max_step = int((max_distance - ray_t0) / step_distance) + 1
    tmpT = ray_t0
    for step in range(max_step):
        if (tmpT + step_distance < max_distance):
            position = ray_origin + ray_direction * tmpT
            sigma = InterpolationField(bb_min, bb_max, extinction_field, position, bb_cell_width)
            accumSigmaXStep += sigma * step_distance
            tmpT += step_distance
        else:
            position = ray_origin + ray_direction * tmpT
            sigma = InterpolationField(bb_min, bb_max, extinction_field, position, bb_cell_width)
            accumSigmaXStep += sigma * (max_distance - tmpT)
            tmpT = max_distance
            break
    return tm.exp(-accumSigmaXStep)

@ti.kernel
def ComputeIntensityKernel(
    lights_num: int,
    light_position: ti.template(),
    light_color: ti.template(),
    lights_distance_attenuation: ti.template(),
    step_distance: float,
    super_samples: int,
    phaseFunc_g: float,
    blackbody_field: ti.template(),
    albedo: tm.vec3,
    extinction_field: ti.template(),
    bb_min: tm.vec3,
    bb_max: tm.vec3,
    bb_cell_width: tm.vec3,
    camera: ti.template(),
    output_intensity_field: ti.template(),
):
    for I in ti.grouped(output_intensity_field):
        spectral_rgb = tm.vec3([ 602.0 * 1.0e-9, 536.0 * 1.0e-9, 448.0 * 1.0e-9 ])
        result = tm.vec3([0.0, 0.0, 0.0])
        for s in range(super_samples):
            position = GetPositionFromDataIndex(tm.vec3(bb_min), tm.vec3(bb_cell_width) , tm.vec3([I[0], I[1], I[2]]), s)
            for light_index in range(lights_num):
                direction = (position - light_position[light_index]).normalized()
                distance = (position - light_position[light_index]).norm()
                intersected_t = IntersectBB(tm.vec3(bb_min) , tm.vec3(bb_max) , tm.vec3(light_position[light_index]), tm.vec3(direction))
                t0 = intersected_t[0]
                t1 = intersected_t[1]
                transmittance = ComputeTransmittanceRay(
                    extinction_field=extinction_field,
                    bb_min=bb_min,
                    bb_max=bb_max,
                    bb_cell_width=bb_cell_width,
                    ray_origin=light_position[light_index],
                    ray_direction=direction,
                    ray_t0=t0,
                    step_distance=step_distance,
                    max_distance=distance
                )
                from_cell_to_eye = (camera.position - position).normalized()
                dot = direction.dot(from_cell_to_eye)
                phaseFunc = 0.5 * (1.0 - phaseFunc_g * phaseFunc_g) / (
                    tm.pow(1.0 + phaseFunc_g * phaseFunc_g - 2.0 * phaseFunc_g * dot, 3.0 / 2))
                emission = tm.vec3([0.0, 0.0, 0.0])
                for i in ti.static(range(3)):
                    emission[i] = (1.0 - albedo[i]) * blackbody_field.GetSpectralRadiance(position, spectral_rgb[i])
                distance_attenuation = 1.0
                if lights_distance_attenuation[light_index] == 1:
                    distance_attenuation = 1.0 / (distance * distance)
                scatter = transmittance * light_color[light_index] * phaseFunc * albedo * distance_attenuation
                result = result + scatter + emission
        output_intensity_field[I[0], I[1], I[2], I[3]] = result[I[3]] / super_samples

def ComputeIntensity(
    density_file_path,
    camera,
    albedo,
    phase_function_g,
    blackbody_field,
    lights, # List[Light],
    step_distance,
    super_samples,
    output_intensity_file_path_r,
    output_intensity_file_path_g,
    output_intensity_file_path_b
):
    density_data_io = DataIO3D.InitLoadFile(density_file_path)
    volume_shape = density_data_io.get_shape()
    volume_bb_min = density_data_io.bb_min
    volume_bb_max_ti = density_data_io.bb_max_taichi()
    volume_bb_cell_width = density_data_io.cell_width
    out_intensity_field = ti.field(dtype=ti.f32, shape=(volume_shape[0], volume_shape[1], volume_shape[2], 3))
    
    lights_num = len(lights)
    lights_position = ti.Vector.field(3, dtype=ti.f32, shape=lights_num)
    lights_color = ti.Vector.field(3, dtype=ti.f32, shape=lights_num)
    lights_distance_attenuation = ti.field(dtype=ti.i16, shape=lights_num)
    lights_position.from_numpy(np.array([light.position for light in lights]))
    lights_color.from_numpy(np.array([light.color for light in lights]))
    lights_distance_attenuation.from_numpy(np.array([int(light.distance_attenuation) for light in lights]))
    
    ComputeIntensityKernel(
        lights_num=lights_num,
        light_position=lights_position,
        light_color=lights_color,
        lights_distance_attenuation=lights_distance_attenuation,
        step_distance=step_distance,
        super_samples=super_samples,
        phaseFunc_g=phase_function_g,
        blackbody_field=blackbody_field,
        albedo=albedo,
        extinction_field=density_data_io.data,
        bb_min=volume_bb_min,
        bb_max=volume_bb_max_ti,
        bb_cell_width=volume_bb_cell_width,
        camera=camera,
        output_intensity_field=out_intensity_field,
    )

    out_intensity_np = out_intensity_field.to_numpy()
    Output3D(volume_bb_min, volume_bb_cell_width, out_intensity_np[:, :, :, 0], output_intensity_file_path_r)
    Output3D(volume_bb_min, volume_bb_cell_width, out_intensity_np[:, :, :, 1], output_intensity_file_path_g)
    Output3D(volume_bb_min, volume_bb_cell_width, out_intensity_np[:, :, :, 2], output_intensity_file_path_b)

def main():
    args = argparse.ArgumentParser()
    args.add_argument('-compute_mode', type=str, required=True, help='cpu or gpu')
    args.add_argument('-extinction', type=str, required=True, help='extinction file path')
    args.add_argument('-xml', type=str, required=True, help='xml file path')
    args.add_argument('-out_r', type=str, required=True, help='output illuminance file path R')
    args.add_argument('-out_g', type=str, required=True, help='output illuminance file path G')
    args.add_argument('-out_b', type=str, required=True, help='output illuminance file path B')
    args.add_argument('-temperature', type=str, help='temperature file path', default=None, const=None, nargs='?')
    args = args.parse_args()
    
    if args.compute_mode == 'cpu':
        ti.init(arch=ti.cpu)
    elif args.compute_mode == 'gpu':
        ti.init(arch=ti.gpu)
    else:
        print("Invalid compute mode. Use 'cpu' or 'gpu'.")
        exit(-1)
    extinction_file_path = args.extinction
    xml_path = args.xml
    output_illuminance_file_path_R = args.out_r
    output_illuminance_file_path_G = args.out_g
    output_illuminance_file_path_B = args.out_b
    temperature_file_path = args.temperature
    
    
    xml = xml_load(xml_path)
    
    phase_function_HG = xml.get_phase_function_HG()
    step_distance = xml.get_step_distance()

    cam = xml.get_camera()
    light = xml.get_light()

    albedo = xml.get_albedo()
    min_temp = xml.get_temp_min()
    max_temp = xml.get_temp_max()
    temp_factor = xml.get_temp_factor()
    blackbody_factor = xml.get_blackbody_factor()
    cell_super_samples = xml.get_cell_super_samples()
    blackbody = BlackBodyField(min_temp, max_temp, temp_factor, blackbody_factor, temperature_file_path)
    ComputeIntensity(
        density_file_path=extinction_file_path,
        camera=cam,
        albedo=albedo,
        phase_function_g=phase_function_HG,
        blackbody_field=blackbody,
        lights=light,
        step_distance=step_distance,
        super_samples=cell_super_samples,
        output_intensity_file_path_r=output_illuminance_file_path_R,
        output_intensity_file_path_g=output_illuminance_file_path_G,
        output_intensity_file_path_b=output_illuminance_file_path_B
    )


if __name__ == '__main__':
    main()
