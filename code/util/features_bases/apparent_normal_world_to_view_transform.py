# -----------------------------------------------------------------------------
# Stroke Transfer for Participating Media
# http://cg.it.aoyama.ac.jp/yonghao/sig25/abstsig25.html
#
# File: apparent_normal_world_to_view_transform.py
# Maintainer: Naoto Shirashima
#
# Description:
# Transforming the world coordinate system normal on the grid to the screen coordinate system
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
from data_io import  DataIO2D, Output2D


@ti.kernel
def CameraDotDirXYZ(camera:ti.template(), apparent_taichi_field: ti.template(), result_taichi_field: ti.template()):
    dir_x, dir_y, dir_z = camera.GetXYZ()
    for I in ti.grouped(result_taichi_field):
        world_normal = tm.vec3(apparent_taichi_field[I[0], I[1], 0], apparent_taichi_field[I[0], I[1], 1], apparent_taichi_field[I[0], I[1], 2])
        local_normal = tm.vec3( world_normal.dot(dir_x), world_normal.dot(dir_y), world_normal.dot(dir_z) )
        result_taichi_field[I[0], I[1], I[2]] = local_normal[I[2]]


def main():
    compute_mode = sys.argv[1]
    if compute_mode == 'cpu':
        ti.init(arch=ti.cpu)
    elif compute_mode == 'gpu':
        ti.init(arch=ti.gpu)
    apparent_file_path = sys.argv[2]
    xml_path = sys.argv[3]
    output_x_file_path = sys.argv[4]
    output_y_file_path = sys.argv[5]
    output_z_file_path = sys.argv[6]

    xml = xml_load(xml_path)
    cam = xml.get_camera()

    apparent_data = DataIO2D.InitLoadFile(apparent_file_path)
    result = ti.field(dtype=ti.f32, shape=(apparent_data.data_np.shape))
    CameraDotDirXYZ(camera=cam, apparent_taichi_field=apparent_data.data, result_taichi_field=result)
    result_np = result.to_numpy()

    Output2D(result_np[:,:,0], output_x_file_path)
    Output2D(result_np[:,:,1], output_y_file_path)
    Output2D(result_np[:,:,2], output_z_file_path)

if __name__ == '__main__':
    main()