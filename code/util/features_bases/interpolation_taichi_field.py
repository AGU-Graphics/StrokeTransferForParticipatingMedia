# -----------------------------------------------------------------------------
# Stroke Transfer for Participating Media
# http://cg.it.aoyama.ac.jp/yonghao/sig25/abstsig25.html
#
# File: interpolation_taichi_field.py
# Maintainer: Naoto Shirashima and Yuki Yamaoka
#
# Description:
# Completion of ti.field for taichi
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
import numpy as np

# The following functions are defined in this file:
# InterpolationField : linear interpolation for taichi field
# CubicBSpline3D : cubic B-spline interpolation for taichi field

@ti.func
def InterpolationField(bb_min: tm.vec3, bb_max: tm.vec3, data_field: ti.template(), position: tm.vec3, cell_width: tm.vec3, clamp_mode:int = 0)-> ti.f32:
    # clamp_mode : 0 -> zero, 1 -> one, 2 -> clamp

    over_access = 0
    if clamp_mode == 2:
        position = tm.clamp(position, bb_min, bb_max)
    resolution = data_field.shape
    position_index = tm.vec3([0.0, 0.0, 0.0])
    for i in ti.static(range(3)):
        position_index[i] = (position[i] - bb_min[i]) / cell_width[i]
    data_index_double = tm.vec3(position_index[0], position_index[1], position_index[2])
    data_index0 = tm.floor(position_index[0], dtype=ti.i32)
    data_index1 = tm.floor(position_index[1], dtype=ti.i32)
    data_index2 = tm.floor(position_index[2], dtype=ti.i32)
    result = ti.f32(0.0)
    over_access = data_index0 < 0 or data_index0 >= resolution[0]-1 or data_index1 < 0 or data_index1 >= resolution[1]-1 or data_index2 < 0 or data_index2 >= resolution[2]-1
    if clamp_mode==0 and over_access == 1:
        result = ti.f32(0.0)
    elif clamp_mode==1 and over_access == 1:
        result = ti.f32(1.0)
    else:
        index0 = tm.vec3([data_index0, data_index1, data_index2])
        index1 = tm.vec3([data_index0+1, data_index1+1, data_index2+1])
        itr = data_index_double - index0
        result = (1.0 - itr[0]) * (1.0 - itr[1]) * (1.0 - itr[2]) * data_field[data_index0, data_index1, data_index2] + \
                    (1.0 - itr[0]) * (1.0 - itr[1]) * itr[2] * data_field[data_index0, data_index1, data_index2+1] + \
                    (1.0 - itr[0]) * itr[1] * (1.0 - itr[2]) * data_field[data_index0, data_index1+1, data_index2] + \
                        (1.0 - itr[0]) * itr[1] * itr[2] * data_field[data_index0, data_index1+1, data_index2+1] + \
                            itr[0] * (1.0 - itr[1]) * (1.0 - itr[2]) * data_field[data_index0+1, data_index1, data_index2] + \
                                itr[0] * (1.0 - itr[1]) * itr[2] * data_field[data_index0+1, data_index1, data_index2+1] + \
                                    itr[0] * itr[1] * (1.0 - itr[2]) * data_field[data_index0+1, data_index1+1, data_index2] + \
                                        itr[0] * itr[1] * itr[2] * data_field[data_index0+1, data_index1+1, data_index2+1]
    return result

@ti.func
def InterpolationFieldVec4(bb_min: tm.vec3, data_field: ti.template(), position: tm.vec3, cell_width: tm.vec3)-> ti.f32:
    resolution = data_field.shape
    position_index = tm.vec3([0.0, 0.0, 0.0])
    for i in ti.static(range(3)):
        position_index[i] = (position[i] - bb_min[i]) / cell_width[i]
    data_index_double = tm.vec3(position_index[0], position_index[1], position_index[2])
    data_index0 = tm.floor(position_index[0], dtype=ti.i32)
    data_index1 = tm.floor(position_index[1], dtype=ti.i32)
    data_index2 = tm.floor(position_index[2], dtype=ti.i32)
    result = tm.vec4([0.0, 0.0, 0.0, 0.0])
    if data_index0 < 0 or data_index0 >= resolution[0]-1 or data_index1 < 0 or data_index1 >= resolution[1]-1 or data_index2 < 0 or data_index2 >= resolution[2]-1:
        pass
    else:
        index0 = tm.vec3([data_index0, data_index1, data_index2])
        index1 = tm.vec3([data_index0+1, data_index1+1, data_index2+1])
        itr = data_index_double - index0

        result = (1.0 - itr[0]) * (1.0 - itr[1]) * (1.0 - itr[2]) * data_field[data_index0, data_index1, data_index2] + \
                    (1.0 - itr[0]) * (1.0 - itr[1]) * itr[2] * data_field[data_index0, data_index1, data_index2+1] + \
                    (1.0 - itr[0]) * itr[1] * (1.0 - itr[2]) * data_field[data_index0, data_index1+1, data_index2] + \
                        (1.0 - itr[0]) * itr[1] * itr[2] * data_field[data_index0, data_index1+1, data_index2+1] + \
                            itr[0] * (1.0 - itr[1]) * (1.0 - itr[2]) * data_field[data_index0+1, data_index1, data_index2] + \
                                itr[0] * (1.0 - itr[1]) * itr[2] * data_field[data_index0+1, data_index1, data_index2+1] + \
                                    itr[0] * itr[1] * (1.0 - itr[2]) * data_field[data_index0+1, data_index1+1, data_index2] + \
                                        itr[0] * itr[1] * itr[2] * data_field[data_index0+1, data_index1+1, data_index2+1]
    return result



@ti.func
def InterpolationField2D(bb_min: tm.vec2, data_field: ti.template(), position: tm.vec2, cell_width: tm.vec2, clamp_mode: int) -> ti.f32:
    resolution = data_field.shape
    position_index = tm.vec2([0.0, 0.0])
    for i in ti.static(range(2)):
        position_index[i] = (position[i] - bb_min[i]) / cell_width[i]
        if clamp_mode == 2:
            position_index[i] = tm.clamp(position_index[i], 0.0, resolution[i] - 1.0)
    data_index_double = tm.vec2(position_index[0], position_index[1])
    data_index0 = int(tm.floor(position_index[0]))
    data_index1 = int(tm.floor(position_index[1]))
    result = ti.f32(0.0)
    out_of_range = False
    if clamp_mode == 0:
        if data_index0 < 0 or data_index0 >= resolution[0] - 1 or data_index1 < 0 or data_index1 >= resolution[1] - 1:
            result = ti.f32(0.0)
            out_of_range = True
    elif clamp_mode == 1:
        if data_index0 < 0 or data_index0 >= resolution[0] - 1 or data_index1 < 0 or data_index1 >= resolution[1] - 1:
            result = ti.f32(1.0)
            out_of_range = True
    if out_of_range == False:
        index0 = tm.vec2([data_index0, data_index1])
        index1 = tm.vec2([data_index0 + 1, data_index1 + 1])
        itr = data_index_double - index0
        result = (1.0 - itr[0]) * (1.0 - itr[1]) * data_field[data_index0, data_index1] + \
                 (1.0 - itr[0]) * itr[1] * data_field[data_index0, data_index1 + 1] + \
                 itr[0] * (1.0 - itr[1]) * data_field[data_index0 + 1, data_index1] + \
                 itr[0] * itr[1] * data_field[data_index0 + 1, data_index1 + 1]
    return result

@ti.kernel
def InterpolationFieldKernel(interpolated_field:ti.template(), in_field:ti.template() , bb_min:tm.vec3, bb_max:tm.vec3 , resolution: int):
    for i, j, k in interpolated_field:
        cell_width = (bb_max - bb_min) / resolution
        position = bb_min + tm.vec3([i, j, k]) * cell_width
        interpolated_field[i, j, k] = InterpolationField(bb_min, bb_max, in_field, position)



@ti.data_oriented
class CubicBSpline3D:
    def __init__(self, in_data: ti.template(), bb_min: tm.vec3, cell_width: np.float32):
        self.data = in_data
        self.bb_min = bb_min
        self.cell_width = cell_width
        if isinstance(cell_width, ti.Vector) and cell_width.n > 1:
            print('dim(cell_width)', cell_width.n)
            raise ValueError("cell_width should be a single float value or a 1D array with 3 elements.")
        self.node_num = in_data.shape
        self.cell_num = self.node_num - tm.vec3([1, 1, 1])
        self.bb_max = bb_min + self.cell_num * tm.vec3([cell_width, cell_width, cell_width])

    @ti.static
    def InitByNumpy(in_data: np.ndarray, bb_min: tm.vec3, cell_width: np.float32):
        ti_data = ti.field(dtype=ti.f32, shape=in_data.shape)
        ti_data.from_numpy(in_data)
        my = CubicBSpline3D(ti_data, bb_min, cell_width)
        return my

    @ti.func
    def GetPositionFromIndex(self, index: tm.vec3)-> tm.vec3:
        return self.bb_min + index * self.cell_width
    
    @ti.func
    def Value(self, in_position: tm.vec3)-> ti.f32:
        lerp_h = (in_position - self.bb_min) / self.cell_width
        raw_index_start = ti.floor(lerp_h).cast(ti.int32) - 1
        raw_index_end = raw_index_start + 3
        raw_index_start = [ti.max(raw_index_start[i], 0) for i in ti.static(range(3))]
        raw_index_end = [ti.min(raw_index_end[i], self.node_num[i] - 1) for i in ti.static(range(3))]
        result = ti.f32(0.0)
        for raw_index_x in range(raw_index_start[0], raw_index_end[0] + 1):
            for raw_index_y in range(raw_index_start[1], raw_index_end[1] + 1):
                for raw_index_z in range(raw_index_start[2], raw_index_end[2] + 1):
                    position = self.GetPositionFromIndex(tm.vec3([raw_index_x, raw_index_y, raw_index_z]))
                    result += self.data[raw_index_x, raw_index_y, raw_index_z] * N((in_position - position)[0] / self.cell_width) * N((in_position - position)[1] / self.cell_width) * N((in_position - position)[2] / self.cell_width)
        return result
    
    @ti.func
    def Gradient(self, in_position: tm.vec3)-> tm.vec3:
        return CubicBSpline3DGradient(self.data, tm.vec3(self.bb_min), self.cell_width, in_position)

    @ti.func
    def Gradient2(self, in_position: tm.vec3)-> tm.mat3:
        return CubicBSpline3DGradient2(self.data, tm.vec3(self.bb_min), self.cell_width, in_position)

@ti.func
def CubicBSpline3DGradient(data: ti.template(), bb_min: tm.vec3, cell_width: float, in_position: tm.vec3)-> tm.vec3:
    lerp_h = (in_position - bb_min) / cell_width
    raw_index_start = ti.floor(lerp_h).cast(ti.int32) - 1
    raw_index_end = raw_index_start + 3
    raw_index_start = [ti.max(raw_index_start[i], 0) for i in ti.static(range(3))]
    raw_index_end = [ti.min(raw_index_end[i], data.shape[i] - 1) for i in ti.static(range(3))]
    result = tm.vec3([0.0, 0.0, 0.0])
    for raw_index_x in range(raw_index_start[0], raw_index_end[0] + 1):
        for raw_index_y in range(raw_index_start[1], raw_index_end[1] + 1):
            for raw_index_z in range(raw_index_start[2], raw_index_end[2] + 1):
                result[0] += data[raw_index_x, raw_index_y, raw_index_z] * NGradient(lerp_h[0] - raw_index_x) * N(lerp_h[1] - raw_index_y) * N(lerp_h[2] - raw_index_z) / cell_width
                result[1] += data[raw_index_x, raw_index_y, raw_index_z] * N(lerp_h[0] - raw_index_x) * NGradient(lerp_h[1] - raw_index_y) * N(lerp_h[2] - raw_index_z) / cell_width
                result[2] += data[raw_index_x, raw_index_y, raw_index_z] * N(lerp_h[0] - raw_index_x) * N(lerp_h[1] - raw_index_y) * NGradient(lerp_h[2] - raw_index_z) / cell_width
    return result

@ti.func
def CubicBSpline3DGradient2(data: ti.template(), bb_min: tm.vec3, cell_width: float, in_position: tm.vec3)-> tm.mat3:
    lerp_h = (in_position - bb_min) / cell_width
    raw_index_start = ti.floor(lerp_h).cast(ti.int32) - 1
    raw_index_end = raw_index_start + 3
    raw_index_start = [ti.max(raw_index_start[i], 0) for i in ti.static(range(3))]
    raw_index_end = [ti.min(raw_index_end[i], data.shape[i] - 1) for i in ti.static(range(3))]
    result = tm.mat3([[0.0, 0.0, 0.0], [0.0, 0.0, 0.0], [0.0, 0.0, 0.0]])
    for raw_index_x in range(raw_index_start[0], raw_index_end[0] + 1):
        for raw_index_y in range(raw_index_start[1], raw_index_end[1] + 1):
            for raw_index_z in range(raw_index_start[2], raw_index_end[2] + 1):
                result[0,0] += data[raw_index_x, raw_index_y, raw_index_z] * NGradient2(lerp_h[0] - raw_index_x) * N(lerp_h[1] - raw_index_y) * N(lerp_h[2] - raw_index_z) / (cell_width ** 2)
                result[0,1] += data[raw_index_x, raw_index_y, raw_index_z] * NGradient(lerp_h[0] - raw_index_x) * NGradient(lerp_h[1] - raw_index_y) * N(lerp_h[2] - raw_index_z) / (cell_width ** 2)
                result[0,2] += data[raw_index_x, raw_index_y, raw_index_z] * NGradient(lerp_h[0] - raw_index_x) * N(lerp_h[1] - raw_index_y) * NGradient(lerp_h[2] - raw_index_z) / (cell_width ** 2)

                result[1,1] += data[raw_index_x, raw_index_y, raw_index_z] * N(lerp_h[0] - raw_index_x) * NGradient2(lerp_h[1] - raw_index_y) * N(lerp_h[2] - raw_index_z) / (cell_width ** 2)
                result[1,2] += data[raw_index_x, raw_index_y, raw_index_z] * N(lerp_h[0] - raw_index_x) * NGradient(lerp_h[1] - raw_index_y) * NGradient(lerp_h[2] - raw_index_z) / (cell_width ** 2)

                result[2,2] += data[raw_index_x, raw_index_y, raw_index_z] * N(lerp_h[0] - raw_index_x) * N(lerp_h[1] - raw_index_y) * NGradient2(lerp_h[2] - raw_index_z) / (cell_width ** 2)
    result[1,0] = result[0,1]
    result[2,0] = result[0,2]
    result[2,1] = result[1,2]
    return result

@ti.func
def CubicBSpline2DGetPositionFromIndex(index: tm.vec2, bb_min: tm.vec2, cell_length: float)-> tm.vec2:
    return bb_min + index * cell_length

@ti.func
def CubicBSpline2DGradient(data: ti.template(), bb_min: tm.vec2, cell_length: float, in_position: tm.vec2)-> tm.vec2:
    lerp_h = (in_position - bb_min) / cell_length
    raw_index_start = ti.floor(lerp_h).cast(ti.int32) - 1
    raw_index_end = raw_index_start + 3
    raw_index_start = [ti.max(raw_index_start[i], 0) for i in ti.static(range(2))]
    raw_index_end = [ti.min(raw_index_end[i], data.shape[i] - 1) for i in ti.static(range(2))]
    result = tm.vec2([0.0, 0.0])
    for raw_index_x in range(raw_index_start[0], raw_index_end[0] + 1):
        for raw_index_y in range(raw_index_start[1], raw_index_end[1] + 1):
            position = CubicBSpline2DGetPositionFromIndex(tm.vec2([raw_index_x, raw_index_y]), bb_min, cell_length)
            result[0] += data[raw_index_x, raw_index_y] * NGradient((in_position[0] - position)[0] / cell_length) * N((in_position - position)[1] / cell_length) / cell_length
            result[1] += data[raw_index_x, raw_index_y] * N((in_position[0] - position)[0] / cell_length) * NGradient((in_position[1] - position)[1] / cell_length) / cell_length
    return result

@ti.kernel
def CubicBSpline2DGradientArrayKernel(resolution: tm.vec2, node_num: tm.vec2, data: ti.template(), result:ti.template(), bb_min: tm.vec2, cell_length: float):
    for I in ti.grouped(result):
        position01 = tm.vec2([I[0] / (resolution[0] - 1.0), I[1] / (resolution[1] - 1.0)])
        position = CubicBSpline2DGetPositionFromIndex(tm.vec2([(node_num[0]-1.0) *  position01[0],  (node_num[1]-1.0) *  position01[1] ]), bb_min, cell_length)
        grad = CubicBSpline2DGradient(data, bb_min, cell_length, position)
        result[I[0], I[1]][0] = grad[0]
        result[I[0], I[1]][1] = grad[1]

def CubicBSpline2DGradientArray(resolution: list[int], data: ti.template(), result: ti.template(), bb_min: tm.vec2, cell_length: float):
    node_num = data.shape
    CubicBSpline2DGradientArrayKernel(resolution, node_num, data, result, bb_min, cell_length)

@ti.data_oriented
class CubicBSpline2D:
    def __init__(self, in_data: np.ndarray, bb_min: tm.vec2, cell_with: np.float32):
        self.data = ti.field(dtype=ti.f32, shape=in_data.shape)
        self.data.from_numpy(in_data)
        self.bb_min = bb_min
        self.cell_length = cell_with
        self.node_num = in_data.shape
        self.cell_num = self.node_num - tm.vec2([1, 1])
        self.bb_max = bb_min + self.cell_num * tm.vec2([cell_with, cell_with])
    
    def GetPositionFromIndex(self, index: tm.vec2) -> tm.vec2:
        return CubicBSpline2DGetPositionFromIndex(index, self.bb_min, self.cell_length)

    @ti.func
    def Value(self, in_position: tm.vec2)-> ti.f32:
        lerp_h = (in_position - self.bb_min) / self.cell_length
        raw_index_start = ti.floor(lerp_h).cast(ti.int32) - 1
        raw_index_end = raw_index_start + 3
        raw_index_start = [ti.max(raw_index_start[i], 0) for i in ti.static(range(2))]
        raw_index_end = [ti.min(raw_index_end[i], self.node_num[i] - 1) for i in ti.static(range(2))]
        result = ti.f32(0.0)
        for raw_index_x in range(raw_index_start[0], raw_index_end[0] + 1):
            for raw_index_y in range(raw_index_start[1], raw_index_end[1] + 1):
                position = self.GetPositionFromIndex(tm.vec2([raw_index_x, raw_index_y]))
                result += self.data[raw_index_y, raw_index_x] * N((in_position - position)[0] / self.cell_length) * N((in_position - position)[1] / self.cell_length)
        return result
    
    @ti.func
    def Gradient(self, in_position: tm.vec2)-> tm.vec2:
        return CubicBSpline2DGradient(self.data, self.bb_min, self.cell_length, in_position)
    
    @ti.func
    def Gradient2(self, in_position: tm.vec2)-> tm.mat2:
        lerp_h = (in_position - self.bb_min) / self.cell_length
        raw_index_start = ti.floor(lerp_h).cast(ti.int32) - 1
        raw_index_end = raw_index_start + 3
        raw_index_start = [ti.max(raw_index_start[i], 0) for i in ti.static(range(2))]
        raw_index_end = [ti.min(raw_index_end[i], self.node_num[i] - 1) for i in ti.static(range(2))]
        result = tm.mat2([[0.0, 0.0], [0.0, 0.0]])
        for raw_index_x in range(raw_index_start[0], raw_index_end[0] + 1):
            for raw_index_y in range(raw_index_start[1], raw_index_end[1] + 1):
                position = self.GetPositionFromIndex(tm.vec2([raw_index_x, raw_index_y]))
                result[0,0] += self.data[raw_index_x, raw_index_y] * NGradient2((in_position[0] - position)[0] / self.cell_length) * N((in_position - position)[1] / self.cell_length) / (self.cell_length ** 2)
                result[0,1] += self.data[raw_index_x, raw_index_y] * NGradient((in_position[0] - position)[0] / self.cell_length) * NGradient((in_position[1] - position)[1] / self.cell_length) / (self.cell_length ** 2)
                result[1,1] += self.data[raw_index_x, raw_index_y] * N((in_position[0] - position)[0] / self.cell_length) * NGradient2((in_position[1] - position)[1] / self.cell_length) / (self.cell_length ** 2)
        result[1,0] = result[0,1]
        return result
    
    @ti.kernel
    def ValueArrayKernel(self, result:ti.template(), resolution:tm.vec2):
        for x,y in result:
            position01 = tm.vec2([x / (resolution[0] - 1), y / (resolution[1] - 1)])
            position = self.GetPositionFromIndex(tm.vec2([(node_num[0]-1) *  position01[0],  (node_num[1]-1) *  position01[1] ]))
            result[y,x] = self.Value(position)

    def ValueArray(self, resolution: list[int]):
        resolution = tm.vec2([resolution[0], resolution[1]])
        result = ti.field(dtype=ti.f32, shape=(resolution[0], resolution[1]))
        node_num = self.node_num
        self.ValueArrayKernel(result, resolution)
        return result.to_numpy()

    def GradientArray(self, resolution: list[int], ):
        result = ti.field(dtype=tm.vec2, shape=(resolution[0], resolution[1]))
        CubicBSpline2DGradientArray(resolution, self.data, result, self.bb_min, self.cell_length)
        return result

    
    # @ti.kernel
    # def Gradient2ArrayKernel(self, resolution:tm.vec2, node_num:tm.vec2 , result:ti.template()):
    #     for I in ti.grouped(result):
    #         position01 = tm.vec2([I[0] / (resolution[0] - 1), I[1] / (resolution[1] - 1)])
    #         position = self.GetPositionFromIndex(tm.vec2([(node_num[0]-1) *  position01[0],  (node_num[1]-1) *  position01[1] ]))
    #         grad = self.Gradient2(position)
    #         result[I[0], I[1], 0, 0] = grad[0,0]
    #         result[I[0], I[1], 0, 1] = grad[0,1]
    #         result[I[0], I[1], 1, 0] = grad[1,0]
    #         result[I[0], I[1], 1, 1] = grad[1,1]
    
    def Gradient2Array(self, resolution: list[int]):
        result = ti.field(dtype=ti.f32, shape=(resolution[0], resolution[1], 2, 2))
        node_num = self.node_num
        CubicBSpline2DGradientArrayKernel(resolution, node_num, self.data, result, self.bb_min, self.cell_length)
        return result


@ti.func
def N(x: ti.f32):
    abs_x = ti.abs(x)
    result = ti.f32(0.0)
    if abs_x >= 2.0:
        result = 0.0
    elif abs_x >= 1.0:
        result = (2.0 - abs_x) ** 3 / 6.0
    else:
        result = 0.5*abs_x**3 - abs_x**2 + 2.0/3.0
    return result

@ti.func
def NGradient(x: ti.f32):
    result = ti.f32(0.0)
    if x >= 2.0:
        result = 0.0
    elif x >= 1.0:
        result = -0.5 * (2.0 - x) ** 2
    elif x >= 0.0:
        result = 1.5 * x ** 2 - 2.0 * x
    elif x >= -1.0:
        result = -1.5 * x ** 2 - 2.0 * x
    elif x >= -2.0:
        result = 0.5 * (2.0 + x) ** 2
    else:
        result = 0.0
    return result

@ti.func
def NGradient2(x: ti.f32):
    result = ti.f32(0.0)
    if x >= 2.0:
        result = 0.0
    elif x >= 1.0:
        result = 2.0 - x
    elif x >= 0.0:
        result = 3.0 * x - 2.0
    elif x >= -1.0:
        result = -3.0 * x - 2.0
    elif x >= -2.0:
        result = 2.0 + x
    else:
        result = 0.0
    return result
