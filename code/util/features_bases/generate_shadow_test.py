# -----------------------------------------------------------------------------
# Stroke Transfer for Participating Media
# http://cg.it.aoyama.ac.jp/yonghao/sig25/abstsig25.html
#
# File: generate_shadow_test.py
# Maintainer: Naoto Shirashima
#
# Description:
# [Precalculation] Generate shadow test from surface model
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
import argparse
import matplotlib.pyplot as plt
import json
import os
import numpy as np
from obj_loader import ObjLoader
from bvh import BVH, obj_to_vertex
from data_io import DataIO3D, Output3D
from xml_load import xml_load
from bounding_box import GetPositionFromDataIndex

@ti.kernel
def ComputeShadowTestKernel(
    out_data: ti.template(),
    lights_nums: int,
    for_light: ti.template(),
    lights_position: ti.template(),
    cell_super_sample: int,
    bb_min: tm.vec3,
    cell_width: tm.vec3,
    bvh: ti.template()
):
    for I in ti.grouped(out_data):
        for light_index in range(lights_nums):
            if for_light[light_index] >= 0:
                for ss in range(cell_super_sample):
                    cell_position = GetPositionFromDataIndex(tm.vec3(bb_min), tm.vec3(cell_width), tm.vec3(I), ss)
                    ray_org = lights_position[for_light[light_index]]
                    ray_dir = tm.normalize(cell_position - ray_org)
                    hit_position, hit_face_index = bvh.Intersect(ray_org, ray_dir)
                    if hit_face_index >= 0:
                        out_data[I] += 1.0 / cell_super_sample

def ComputeShadowTest(input_volume, xml, bvh, light_index, output_volume_path):
    cell_super_sample = xml.get_cell_super_samples()
    out = DataIO3D.InitTemplate(input_volume.bb_min, input_volume.cell_width, input_volume.data_np.shape, ti.field(dtype=ti.f32, shape=(input_volume.data_np.shape[0], input_volume.data_np.shape[1], input_volume.data_np.shape[2])))
    lights = xml.get_light()
    lights_nums = len(lights)
    lights_position = ti.Vector.field(3, dtype=ti.f32, shape=lights_nums)
    lights_position.from_numpy(np.array([light.position for light in lights]))
    for_light = ti.field(dtype=ti.i32, shape=(lights_nums))
    for_light_np = np.array([])
    if light_index == -1:
        for_light_np = np.arange(lights_nums)
    else:
        for_light_np = np.append(for_light_np, light_index)
        for i in range(lights_nums-1):
            for_light_np = np.append(for_light_np, -1)
    for_light.from_numpy(np.array(for_light_np, dtype=np.int32))
    
    ComputeShadowTestKernel(
        out_data=out.data,
        lights_nums=lights_nums,
        for_light=for_light,
        lights_position=lights_position,
        cell_super_sample=cell_super_sample,
        bb_min=tm.vec3(input_volume.bb_min),
        cell_width=tm.vec3(input_volume.cell_width),
        bvh=bvh
    )
    result_np = out.data.to_numpy()
    Output3D(out.bb_min, out.cell_width, result_np, output_volume_path)
    

def main():
    ti.init(arch=ti.gpu)
    
    arg = argparse.ArgumentParser()
    arg.add_argument('-i', '--input', type=str, required=True, help='input volume data')
    arg.add_argument('-xml', type=str, required=True, help='input xml file')
    arg.add_argument('-json', type=str, required=True, help='input json file')
    arg.add_argument('-obj', type=str, required=True, help='input obj file')
    arg.add_argument('-c', '--compute_mode', type=str, default='gpu', help='compute mode (cpu or gpu)')
    # arg.add_argument('-light', type=int, default=-1, help='select light index')
    arg.add_argument('-n', '--name', type=str, required=True, help='scene name')
    # arg.add_argument('-o', '--output', type=str, required=True, help='output volume data')

    args = arg.parse_args()
    input_file_path = args.input
    xml_file_path = args.xml
    obj_file_path = args.obj
    compute_mode = args.compute_mode
    # light_index = args.light
    scene_name = args.name
    # output_file_path = args.output
    
    if compute_mode == 'cpu':
        ti.init(arch=ti.cpu)
    elif compute_mode == 'gpu':
        ti.init(arch=ti.gpu)
    else:
        raise ValueError("Invalid compute mode. Choose 'cpu' or 'gpu'.")
    

    # load json
    output_file_paths = []
    with open(args.json, 'r') as f:
        json_data = json.load(f)
        shadow_test_paths = json_data['internal_file_templates']['shadow_test']
        for path in shadow_test_paths:
            output_file_paths.append( os.path.join(scene_name, path) )
    print('output_file_paths', output_file_paths)

    xml = xml_load(xml_file_path)
    object = ObjLoader(obj_file_path)
    vertex = obj_to_vertex(object)
    bvh = BVH(vertex)
    

    
    input_volume = DataIO3D.InitLoadFile(input_file_path)

    for light_index in range(len(output_file_paths)):
        ComputeShadowTest(input_volume=input_volume, xml=xml, bvh=bvh, light_index=light_index, output_volume_path=output_file_paths[light_index])


if __name__ == '__main__':
    main()
