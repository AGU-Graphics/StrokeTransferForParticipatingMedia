# -----------------------------------------------------------------------------
# Stroke Transfer for Participating Media
# http://cg.it.aoyama.ac.jp/yonghao/sig25/abstsig25.html
#
# File: generate_sky_density_from_cloud.py
# Maintainer: Naoto Shirashima
#
# Description:
# Generate density fields for cloud scenes
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
import xml.etree.ElementTree as ET
import numpy as np
from data_io import DataIO3D, Output3D
from interpolation_taichi_field import InterpolationField

@ti.kernel
def InterpolateSphereKernel(whole_density_taichi: ti.template(), outside_density_taichi:ti.template(), whole_density_bb_min:tm.vec3, whole_density_bb_max:tm.vec3, whole_density_bb_cell_width:float, data1_taichi:ti.template(), data1_bb_min:tm.vec3, data1_bb_max:tm.vec3, data1_cell_width:float, outside_sigma_k:ti.f32, offset_sphere:tm.vec3, inner_sphere_radius:tm.vec3, outer_sphere_const:float):
    for I in ti.grouped(whole_density_taichi):
        position = tm.vec3(I[0] *whole_density_bb_cell_width, I[1]*whole_density_bb_cell_width, I[2]*whole_density_bb_cell_width) + whole_density_bb_min
        sphere_sigma = 0.0
        inner_sphere_formula = (position[0]-offset_sphere[0])**2/inner_sphere_radius[0]**2 + (position[1]-offset_sphere[1])**2/inner_sphere_radius[1]**2 + (position[2]-offset_sphere[2])**2/inner_sphere_radius[2]**2
        if inner_sphere_formula <= 1.0:
            sphere_sigma = outside_sigma_k
        elif inner_sphere_formula <= outer_sphere_const:
            # in outer_sphere_formula==1.0, sigma is 0.0
            # in inner_sphere_formula==1.0, sigma is outside_sigma_k
            # so, sigma is linearly interpolated
            itr01 = 1 - (1 - inner_sphere_formula) / (1.0 - outer_sphere_const)
            sphere_sigma = itr01 * outside_sigma_k
        else:
            sphere_sigma = 0.0
        whole_density_taichi[I] = InterpolationField(data1_bb_min, data1_bb_max, data1_taichi, position, tm.vec3(data1_cell_width, data1_cell_width, data1_cell_width)) + sphere_sigma
        outside_density_taichi[I] = sphere_sigma


def string_to_numpy_array(string):
    return np.array(list(map(float, string.split())))

def sphere_sigma():
    input_compute_mode = sys.argv[1]
    if input_compute_mode == 'cpu':
        ti.init(arch=ti.cpu)
    elif input_compute_mode == 'gpu':
        ti.init(arch=ti.gpu)
    else:
        raise ValueError("Invalid compute mode. Use 'cpu' or 'gpu'.")
    input_main_density = sys.argv[2]
    input_xml = sys.argv[3]
    output_hdf5_file = sys.argv[4]
    output_outside_hdf5_file = sys.argv[5]

    outside_sigma_k = ET.parse(input_xml).getroot().find(f'media2').attrib['sigma_k']
    outside_sigma_k = float(outside_sigma_k)

    offset_sphere = string_to_numpy_array( ET.parse(input_xml).getroot().find(f'media2').attrib['offset_sphere'] )
    inner_sphere_radius = string_to_numpy_array( ET.parse(input_xml).getroot().find(f'media2').attrib['inner_sphere_radius'] )
    outer_sphere_constant = float(ET.parse(input_xml).getroot().find(f'media2').attrib['outer_sphere_constant'] )
    outer_sphere_radius = inner_sphere_radius * outer_sphere_constant
    
    print('offset_sphere', offset_sphere)

    density = DataIO3D.InitLoadFile(input_main_density)
    cell_width = density.cell_width[0]
    print('data 1 bb_min', density.bb_min)
    print('data 1 bb_max', density.bb_max_python())
    print('data 1 bb_size', density.bb_max_python() - density.bb_min)
    print('data 1 center', (density.bb_max_python() + density.bb_min) / 2)

    output_bb_min = -outer_sphere_radius + offset_sphere
    output_bb_min = np.minimum(output_bb_min, density.bb_min)
    print('output_bb_min', output_bb_min)
    output_bb_max = outer_sphere_radius + offset_sphere
    output_bb_max = np.maximum(output_bb_max, density.bb_max_python())
    print('output_bb_max', output_bb_max)
    output_resolution = np.array([ (output_bb_max-output_bb_min)[0]/cell_width, (output_bb_max-output_bb_min)[1]/cell_width, (output_bb_max-output_bb_min)[2]/cell_width ])
    output_resolution = np.ceil(output_resolution).astype(np.int32)

    print('output_resolution', output_resolution)

    whole_density_taichi = ti.field(ti.f32, shape=(output_resolution[0], output_resolution[1], output_resolution[2]))
    outside_density_taichi = ti.field(ti.f32, shape=(output_resolution[0], output_resolution[1], output_resolution[2]))
    whole_density_bb_min = output_bb_min
    whole_density_bb_max = output_bb_max
    whole_density_bb_min = tm.vec3(whole_density_bb_min)
    whole_density_bb_max = tm.vec3(whole_density_bb_max)
    InterpolateSphereKernel(
        whole_density_taichi=whole_density_taichi,
        outside_density_taichi=outside_density_taichi,
        whole_density_bb_min=whole_density_bb_min,
        whole_density_bb_max=whole_density_bb_max,
        whole_density_bb_cell_width=cell_width,
        data1_taichi=density.data,
        data1_bb_min=density.bb_min,
        data1_bb_max=density.bb_max_taichi(),
        data1_cell_width=cell_width,
        outside_sigma_k=outside_sigma_k,
        offset_sphere=offset_sphere,
        inner_sphere_radius=inner_sphere_radius,
        outer_sphere_const=outer_sphere_constant
        )

    Output3D(whole_density_bb_min, cell_width, whole_density_taichi.to_numpy(), output_hdf5_file)
    Output3D(whole_density_bb_min, cell_width, outside_density_taichi.to_numpy(), output_outside_hdf5_file)



if __name__ == '__main__':
    sphere_sigma()
