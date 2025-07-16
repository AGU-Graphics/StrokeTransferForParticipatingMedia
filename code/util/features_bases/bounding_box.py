# -----------------------------------------------------------------------------
# Stroke Transfer for Participating Media
# http://cg.it.aoyama.ac.jp/yonghao/sig25/abstsig25.html
#
# File: bounding_box.py
# Maintainer: Naoto Shirashima
#
# Description:
# Collision calculation with bounding box rays, conversion from lattice index to world coordinates
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
from sample_points_generator import halton_sequence_3d

@ti.func
def IntersectBB(bb_min:tm.vec3, bb_max:tm.vec3 , ray_org: tm.vec3, ray_dir: tm.vec3) -> list[ti.f32]:
    inv_dir = tm.vec3( 1.0 / ray_dir[0], 1.0 / ray_dir[1], 1.0 / ray_dir[2])
    t = ti.Vector([0., 0., 0., 0., 0., 0.])
    t[0] = (bb_min[0] - ray_org[0]) * inv_dir[0]
    t[1] = (bb_min[1] - ray_org[1]) * inv_dir[1]
    t[2] = (bb_min[2] - ray_org[2]) * inv_dir[2]
    t[3] = (bb_max[0] - ray_org[0]) * inv_dir[0]
    t[4] = (bb_max[1] - ray_org[1]) * inv_dir[1]
    t[5] = (bb_max[2] - ray_org[2]) * inv_dir[2]
    sel0 = int(ray_dir[0] < 0.0)
    sel1 = int(ray_dir[1] < 0.0)
    sel2 = int(ray_dir[2] < 0.0)
    near = max(t[sel0*3+0], t[sel1*3+1], t[sel2*3+2])
    far = min(t[(1-sel0)*3+0], t[(1-sel1)*3+1], t[(1-sel2)*3+2])
    return [max(near,0.0), max(far, 0.0)]

@ti.func
def GetPositionFromDataIndex(bb_min:tm.vec3 , cell_width:tm.vec3 , dataindex: tm.vec3, cell_super_sample: int) -> tm.vec3:
    random_number_01 = halton_sequence_3d(cell_super_sample)
    offset_index = random_number_01 - tm.vec3([0.5, 0.5, 0.5])
    res = tm.vec3([0.0, 0.0, 0.0])
    for i in ti.static(range(3)):
        res[i] = (dataindex[i] + offset_index[i]) * cell_width[i] + bb_min[i]
    return res

@ti.func
def ClampPositionByBB(position:tm.vec3, bb_min:tm.vec3, bb_max:tm.vec3):
    return tm.clamp(position, bb_min, bb_max)
