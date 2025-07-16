# -----------------------------------------------------------------------------
# Stroke Transfer for Participating Media
# http://cg.it.aoyama.ac.jp/yonghao/sig25/abstsig25.html
#
# File: convert_world_velocity_to_screen_space.py
# Maintainer: Naoto Shirashima
#
# Description:
# Transform the velocity field in the world coordinate system on the grid into a velocity field relative to the screen
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
from data_io import DataIO3D
from xml_load import xml_load
from data_io import Output3D
from camera import Camera
from bounding_box import GetPositionFromDataIndex

@ti.data_oriented
class ConvertWorld2ScreenSpace:
    def __init__(self, velocity_file_path, velocity_x_path, velocity_y_path, velocity_z_path, camera1: Camera, camera2: Camera, fps: float):
        input_h5 = DataIO3D.InitLoadFile(velocity_file_path)
        self.bb_min = input_h5.bb_min
        self.bb_max = input_h5.bb_max_taichi()
        self.cell_width = input_h5.cell_width
        self.cell_width = tm.vec3([self.cell_width[0], self.cell_width[0], self.cell_width[0]])
        self.shape = input_h5.get_shape()
        self.data = input_h5.data
        resolution = self.shape

        self.camera1 = camera1
        self.camera2 = camera2
        self.fps = fps

        self.dt = 1.0 / self.fps

        velocity_x = DataIO3D.InitLoadFile(velocity_x_path, taichi=False, checkSumTest=True)
        velocity_y = DataIO3D.InitLoadFile(velocity_y_path, taichi=False, checkSumTest=True)
        velocity_z = DataIO3D.InitLoadFile(velocity_z_path, taichi=False, checkSumTest=True)
        self.velocity_x = ti.field(ti.f32, shape=(resolution[0], resolution[1], resolution[2]))
        self.velocity_y = ti.field(ti.f32, shape=(resolution[0], resolution[1], resolution[2]))
        self.velocity_z = ti.field(ti.f32, shape=(resolution[0], resolution[1], resolution[2]))

        self.velocity_x.from_numpy(velocity_x.data_np)
        self.velocity_y.from_numpy(velocity_y.data_np)
        self.velocity_z.from_numpy(velocity_z.data_np)

        self.result = ti.field(ti.f32, shape=(self.shape[0], self.shape[1], self.shape[2], 2))
        self.result.fill(0.0)

    @ti.func
    def Convertor(self, super_samples: int):
        dt = self.dt
        for x, y, z in self.data:
            for ss in range(super_samples):
                cell_position = GetPositionFromDataIndex(tm.vec3(self.bb_min), tm.vec3(self.cell_width) , tm.vec3([x, y, z]), ss)
                velocity_start_position = cell_position
                world_velocity = tm.vec3([self.velocity_x[x, y, z], self.velocity_y[x, y, z], self.velocity_z[x, y, z]])
                velocity_end_position = velocity_start_position + world_velocity * dt
                start_uv = self.camera1.ComputeNonDimensionalizedAndCenteredUVFromPosition(velocity_start_position)
                end_uv = self.camera2.ComputeNonDimensionalizedAndCenteredUVFromPosition(velocity_end_position)
                if tm.isnan(start_uv[0]) or tm.isnan(start_uv[1]) or tm.isnan(end_uv[0]) or tm.isnan(end_uv[1]):
                    continue
                uv_direction_in_raytracing_space = end_uv - start_uv
                uv_direction_in_raytracing_space = uv_direction_in_raytracing_space / dt
                uv_direction_in_screen_space = tm.vec2([uv_direction_in_raytracing_space[0], uv_direction_in_raytracing_space[1]])
                self.result[x, y, z, 0] += uv_direction_in_screen_space[0]
                self.result[x, y, z, 1] += uv_direction_in_screen_space[1]
            self.result[x, y, z, 0] /= super_samples
            self.result[x, y, z, 1] /= super_samples

    @ti.kernel
    def ComputePositionToUV_Kernel(self , super_samples: int, result_x: ti.template(), result_y: ti.template()):
        self.Convertor(super_samples)
        for i, j, k in result_x:
            result_x[i, j, k] = self.result[i, j, k, 0]
            result_y[i, j, k] = self.result[i, j, k, 1]

    def Compute(self, super_samples: int):
        result_x = ti.field(ti.f32, shape=(self.shape[0], self.shape[1], self.shape[2]))
        result_y = ti.field(ti.f32, shape=(self.shape[0], self.shape[1], self.shape[2]))
        self.ComputePositionToUV_Kernel(super_samples, result_x, result_y)
        return result_x, result_y


def main():
    print('main')
    compute_mode = sys.argv[1]
    if compute_mode == 'cpu':
        ti.init(arch=ti.cpu)
    elif compute_mode == 'gpu':
        ti.init(arch=ti.gpu)

    density_file_path = sys.argv[2]
    velocity_x = sys.argv[3]
    velocity_y = sys.argv[4]
    velocity_z = sys.argv[5]
    xmlPath = sys.argv[6]
    xmlPath2 = sys.argv[7]
    fps = float(sys.argv[8])

    result_u_file_path = sys.argv[9]
    result_v_file_path = sys.argv[10]
    xml1 = xml_load(xmlPath)
    xml2 = xml_load(xmlPath2)
    cam1 = xml1.get_camera()
    cam2 = xml2.get_camera()

    cell_super_samples = xml1.get_cell_super_samples()

    convert_world_to_screen_space = ConvertWorld2ScreenSpace(density_file_path, velocity_x, velocity_y, velocity_z, cam1, cam2, fps)

    result_u, result_v = convert_world_to_screen_space.Compute(cell_super_samples)

    result_u = result_u.to_numpy()
    result_v = result_v.to_numpy()

    Output3D(convert_world_to_screen_space.bb_min, convert_world_to_screen_space.cell_width, result_u, result_u_file_path)
    Output3D(convert_world_to_screen_space.bb_min, convert_world_to_screen_space.cell_width, result_v, result_v_file_path)

if __name__ == '__main__':
    main()
