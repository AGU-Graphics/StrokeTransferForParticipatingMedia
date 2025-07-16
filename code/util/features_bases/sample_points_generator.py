# -----------------------------------------------------------------------------
# Stroke Transfer for Participating Media
# http://cg.it.aoyama.ac.jp/yonghao/sig25/abstsig25.html
#
# File: sample_points_generator.py
# Maintainer: Naoto Shirashima
#
# Description:
# Generate quasi-random points
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


@ti.func
def halton_sequence(index: int, base: int)-> ti.f32:
    result = ti.f32(0.0)
    f = 1.0 / base
    i = index + 1
    while i > 0:
        result = result + f * (i % base)
        i = i // base
        f = f / base
    return result

@ti.func
def halton_sequence_2d(index: int)-> tm.vec3:
    return tm.vec2([halton_sequence(index, 2), halton_sequence(index, 3)])

@ti.func
def halton_sequence_3d(index: int)-> tm.vec3:
    return tm.vec3([halton_sequence(index, 2), halton_sequence(index, 3), halton_sequence(index, 5)])

@ti.func
def vu_point_to_uv_point_with_haltonseq(input_vu, super_sample):
    random_number_01 = halton_sequence_2d(super_sample)
    offset_index = random_number_01
    uv_position_index = tm.vec2([input_vu[1], input_vu[0]])
    uv = tm.vec2([uv_position_index[0] + offset_index[0] - 0.5, uv_position_index[1] + offset_index[1] - 0.5])
    return uv
