# -----------------------------------------------------------------------------
# Stroke Transfer for Participating Media
# http://cg.it.aoyama.ac.jp/yonghao/sig25/abstsig25.html
#
# File: integral_conversion.py
# Maintainer: Naoto Shirashima and Yuki Yamaoka
#
# Description:
# A component that performs line-of-sight integration for any volume data.
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
from camera import Camera
from depth_value import DepthValue


@ti.kernel
def ComputeScreenIntegral1Kernel(screen_data: ti.template(), step_distance: float, camera: ti.template(), depth_value: ti.template(),
                                super_samples: int, integrand_func: ti.template()):
    for I in ti.grouped(screen_data):
        result = 0.0
        depth = tm.inf
        for ss in range(super_samples):
            org, dir = camera.OrgDirFromVU(I, ss)
            if depth_value.is_valid():
                vu = tm.vec2([I[0], I[1]])
                uv_halton_seq = vu_point_to_uv_point_with_haltonseq(vu, ss)
                uv01_halton_seq = uv_halton_seq / tm.vec2([camera.pixels[0], camera.pixels[1]])
                depth = depth_value.GetDepth01(vu_01=tm.vec2([uv01_halton_seq[1], uv01_halton_seq[0]]))
            result += integrand_func(org, dir, step_distance, depth)

        screen_data[I[0], I[1]] = result / super_samples

@ti.kernel
def ComputeScreenIntegral3Kernel(screen_data: ti.template(), step_distance: float, camera: ti.template(), depth_value: ti.template(),
                                super_samples: int, integrand_func: ti.template()):
    for I in ti.grouped(screen_data):
        result = tm.vec3([0.0, 0.0, 0.0])
        depth = tm.inf
        for ss in range(super_samples):
            org, dir = camera.OrgDirFromVU(tm.vec2([I[0], I[1]]), ss)
            if depth_value.is_valid():
                vu = tm.vec2([I[0], I[1]])
                uv_halton_seq = vu_point_to_uv_point_with_haltonseq(vu, ss)
                uv01_halton_seq = uv_halton_seq / tm.vec2([camera.pixels[0], camera.pixels[1]])
                depth = depth_value.GetDepth01(vu_01=tm.vec2([uv01_halton_seq[1], uv01_halton_seq[0]]))
            result += integrand_func(org, dir, step_distance, depth)
        screen_data[I[0], I[1], I[2]] = result[I[2]] / super_samples

@ti.kernel
def ComputeScreenIntegralKernel(screen_data: ti.template(), step_distance: float, camera: ti.template(), depth_value: ti.template(),
                                super_samples: int, integrand_func: ti.template(), dim: int):
    for I in ti.grouped(screen_data):
        # resultを、ti.Vectorのdim次元のベクトルにする
        result = ti.Vector.zero(n=dim, dt=ti.f32)
        depth = tm.inf
        for ss in range(super_samples):
            org, dir = camera.OrgDirFromVU(tm.vec2([I[0], I[1]]), ss)
            if depth_value.is_valid():
                vu = tm.vec2([I[0], I[1]])
                uv_halton_seq = vu_point_to_uv_point_with_haltonseq(vu, ss)
                uv01_halton_seq = uv_halton_seq / tm.vec2([camera.pixels[0], camera.pixels[1]])
                depth = depth_value.GetDepth01(vu_01=tm.vec2([uv01_halton_seq[1], uv01_halton_seq[0]]))
            result += integrand_func(org, dir, step_distance, depth)
        screen_data[I[0], I[1], I[2]] = result[I[2]] / super_samples

def ComputeScreenIntegral(dim: int, step_distance: float, camera: Camera, depth_value: DepthValue, super_samples: int,
                            integrand_func: classmethod):
    if dim == 1:
        screen_data = ti.field(dtype=ti.f32, shape=(camera.pixels[1], camera.pixels[0]))
        ComputeScreenIntegral1Kernel(screen_data=screen_data, step_distance=step_distance, camera=camera,
                                    depth_value=depth_value, super_samples=super_samples,
                                    integrand_func=integrand_func)
    elif dim == 3:
        screen_data = ti.field(dtype=ti.f32, shape=(camera.pixels[1], camera.pixels[0], 3))
        ComputeScreenIntegral3Kernel(screen_data=screen_data, step_distance=step_distance, camera=camera,
                                    depth_value=depth_value, super_samples=super_samples,
                                    integrand_func=integrand_func)
    else:
        screen_data = ti.field(dtype=ti.f32, shape=(camera.pixels[1], camera.pixels[0], dim))
        ComputeScreenIntegralKernel(screen_data=screen_data, step_distance=step_distance, camera=camera,
                                    depth_value=depth_value, super_samples=super_samples,
                                    integrand_func=integrand_func, dim=dim)
    return screen_data
