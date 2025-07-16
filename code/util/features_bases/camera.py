# -----------------------------------------------------------------------------
# Stroke Transfer for Participating Media
# http://cg.it.aoyama.ac.jp/yonghao/sig25/abstsig25.html
#
# File: camera.py
# Maintainer: Naoto Shirashima
#
# Description:
# Camera Class
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
from sample_points_generator import vu_point_to_uv_point_with_haltonseq

class Camera:
    def __init__(self, position: tm.vec3, direction: tm.vec3, pixels: list[int], film: list[float], fov2_radian: list[float]):
        self.position = position
        self.direction = direction
        self.up = tm.vec3([0, 1, 0])
        self.pixels = pixels
        sensor_width = film[0]
        sensor_height = film[1]
        self.film = film
        self.fov2_radian = fov2_radian
        if sensor_width >= sensor_height:
            fovy_rad = fov2_radian[0]
            self.pixels = tm.vec2([pixels[0], pixels[0] / sensor_width * sensor_height])
            self.fovxHalfLength = tm.tan(fovy_rad / 2.0)
            self.fovyHalfLength = self.fovxHalfLength / sensor_width * sensor_height
        else:
            fovy_rad = fov2_radian[1]
            self.pixels = tm.vec2([pixels[1] / sensor_height * sensor_width, pixels[1]])
            self.fovyHalfLength = tm.tan(fovy_rad / 2.0)
            self.fovxHalfLength = self.fovyHalfLength / sensor_height * sensor_width
        
    @ti.func
    def GetXYZ(self)-> tm.vec3:
        z = -tm.normalize(self.direction)
        _y = tm.vec3([0, 1, 0])
        y = tm.normalize(_y - tm.dot(z, _y) * z)
        x = tm.cross(y, z)
        return x, y, z
    
    @ti.func
    def OrgDirFromVU(self, input_vu_data_index: tm.ivec2, super_sample: int):
        uv = vu_point_to_uv_point_with_haltonseq(input_vu_data_index, super_sample)
        org = self.position
        dir = self.DirectionFromUV(tm.vec2([uv[0], uv[1]]))
        return org, dir

    @ti.func
    def DirectionFromUV(self, uv: tm.vec2)-> tm.vec3:
        uv01 = uv / tm.vec2([self.pixels[0] -1 , self.pixels[1] - 1])
        dirScreenButtomLeft = tm.vec3([-self.fovxHalfLength, -self.fovyHalfLength, -1.0])
        dirScreenUpperRight = tm.vec3([self.fovxHalfLength, self.fovyHalfLength, -1.0])
        dir = dirScreenButtomLeft + (dirScreenUpperRight - dirScreenButtomLeft) * (tm.vec3([uv01[0], uv01[1], 0.0]))
        dir = tm.normalize(dir)
        x,y,z = self.GetXYZ()
        dirWorld = x * dir[0] + y * dir[1] + z * dir[2]
        return tm.normalize(dirWorld)

    @ti.func
    def ComputeNonDimensionalizedAndCenteredUVFromDirection(self, direction: tm.vec3)-> tm.vec2:
        fovyHalfLength = self.fovyHalfLength
        fovxHalfLength = self.fovxHalfLength
        result_uv = tm.vec2([tm.nan, tm.nan])
        # C++のGetDirXYZの内容
        # z = -self.direction.normalize()
        z = -tm.normalize(self.direction)
        _y = tm.vec3([0, 1, 0])
        # y = tm.cross(_y, z).normalize()
        y = tm.normalize(_y - tm.dot(z, _y) * z)
        x = tm.cross(y, z)

        local_dir = tm.vec3([tm.dot(direction, x), tm.dot(direction, y), tm.dot(direction, z)])
        # ignore the case that input direction is behind the camera
        if -local_dir[2] > 0.05:
            screen_position = tm.vec2([local_dir[0], local_dir[1]])/(-local_dir[2])

            dirScreenBottomLeft = tm.vec2([-fovxHalfLength, -fovyHalfLength])
            dirScreenUpperRight = tm.vec2([fovxHalfLength, fovyHalfLength])

            tmp_max_element = dirScreenUpperRight - dirScreenBottomLeft
            # tmp_max = tm.max(tmp_max_element[0], tmp_max_element[1])
            tmp_max = tmp_max_element[0]
            if tmp_max_element[1] > tmp_max_element[0]:
                tmp_max = tmp_max_element[1]
            else:
                tmp_max = tmp_max_element[0]
            result_uv = tm.vec2([screen_position[0] / tmp_max,  screen_position[1] / tmp_max])
        return result_uv
    
    @ti.func
    def ComputeNonDimensionalizedAndCenteredUVFromPosition(self, position: tm.vec3)-> tm.vec2:
        direction = position - self.position
        return self.ComputeNonDimensionalizedAndCenteredUVFromDirection(direction)