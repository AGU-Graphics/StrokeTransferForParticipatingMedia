# -----------------------------------------------------------------------------
# Stroke Transfer for Participating Media
# http://cg.it.aoyama.ac.jp/yonghao/sig25/abstsig25.html
#
# File: generate_depth_value.py
# Maintainer: Naoto Shirashima
#
# Description:
# [Precalculation] Generate depth values from surface models
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
import numpy as np
import taichi as ti
import taichi.math as tm
import argparse
import os
from tqdm import tqdm
from obj_loader import ObjLoader
from xml_load import xml_load
from camera import Camera
from data_io import DataIO2D, Output2D

@ti.kernel
def ComputeDepthTestKernel(
        object: ti.template(),
        super_samples: int,
        camera: ti.template(),
        output_screen_data: ti.template(),
    ):
    for I in ti.grouped(output_screen_data):
        result = 0.0
        for ss in range(super_samples):
            org, dir = camera.OrgDirFromVU(I, ss)
            ray_t, intersect_face_index = object.IntersectOBJ(org, dir)
            if ray_t < -1:
                result += tm.inf
            else:
                result += ray_t
        output_screen_data[I] = result


def main():
    arg = argparse.ArgumentParser()
    arg.add_argument('--obj', type=str, required=True, help='input obj file')
    arg.add_argument('--xml', type=str, required=True, help='input xml file')
    arg.add_argument('--plot', action='store_true', help='plot depth map')
    arg.add_argument('-o', '--out', type=str, required=True, help='output data')
    arg.add_argument('--res', type=int, default=1, help='multi resolution')
    arg.add_argument('--compute_mode', type=str, default='gpu', help='compute mode (cpu or gpu)')
    arg.add_argument('--screen_size_multi', type=float, default=1.0, help='multi screen size')
    arg.add_argument('-s', type=int, required=True, help='start frame number')
    arg.add_argument('-e', type=int, required=True, help='end frame number')
    arg.add_argument('--skip', type=int, default=1, help='frame skip')
    args = arg.parse_args()
    object_file_name = args.obj
    xml_file = args.xml
    out_file = args.out
    plot = args.plot
    res = args.res
    compute_mode = args.compute_mode
    screen_size_multi = args.screen_size_multi
    frame_start = args.s
    frame_end = args.e
    frame_skip = args.skip
    plot = args.plot
    
    if compute_mode == 'cpu':
        ti.init(arch=ti.cpu)
    elif compute_mode == 'gpu':
        ti.init(arch=ti.gpu)
    else:
        raise ValueError("Invalid compute mode. Use 'cpu' or 'gpu'.")
    
    for frame_index in tqdm(range(frame_start, frame_end + 1, frame_skip)):
        if '%' in object_file_name:
            tmp_object_file_name = object_file_name % frame_index
        else:
            tmp_object_file_name = object_file_name
        object = ObjLoader(tmp_object_file_name)

        if '%' in xml_file:
            tmp_xml_file = xml_file % frame_index
        else:
            tmp_xml_file = xml_file
        xml = xml_load(tmp_xml_file)
        camera = xml.get_camera()
        cam_pix = camera.pixels
        cam_pix *= res
        # camera.pixels = cam_pix
        # print('cam_pix', cam_pix)
        # # multi screen size
        fov = camera.fov2_radian
        # # print('fov', fov)
        screen_size_0 = np.tan(fov[0] * 0.5) * 2.0
        screen_szie_1 = np.tan(fov[1] * 0.5) * 2.0
        # # print('screen_size', screen_size_0, screen_szie_1)
        screen_size_0 *= screen_size_multi
        screen_szie_1 *= screen_size_multi
        # # print('screen_size', screen_size_0, screen_szie_1)
        fov[0] = np.arctan(screen_size_0 * 0.5) * 2.0
        fov[1] = np.arctan(screen_szie_1 * 0.5) * 2.0
        # # print('fov', fov)
        new_cam = Camera(camera.position, camera.direction, cam_pix, camera.film, fov)
        screen_data = DataIO2D.TaichiFieldInit([cam_pix[1].astype(int), cam_pix[0].astype(int)], dtype=ti.f32)

        ComputeDepthTestKernel(
            object=object,
            super_samples=1,
            camera=new_cam,
            output_screen_data=screen_data
        )
        result_np = screen_data.to_numpy()
        if '%' in out_file:
            tmp_out_file = out_file % frame_index
        else:
            tmp_out_file = out_file
        os.makedirs(os.path.dirname(tmp_out_file), exist_ok=True)
        Output2D(result_np, tmp_out_file)
        
        if plot:
            import matplotlib.pyplot as plt
            plt.imshow(result_np, origin='lower')
            plt.colorbar()
            plt.title(f'Depth Map Frame {frame_index}')
            print('output', tmp_out_file)
            print('dir', os.path.dirname(tmp_out_file))
            print('file', os.path.basename(tmp_out_file))
            print('file name', os.path.basename(tmp_out_file).split('.')[0])
            out_png_path = os.path.join( os.path.dirname(tmp_out_file), f'{os.path.basename(tmp_out_file).split(".")[0]}.png' )
            plt.savefig(out_png_path)
            plt.close()



if __name__ == '__main__':
    main()

