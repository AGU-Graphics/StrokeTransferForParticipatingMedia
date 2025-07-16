# -----------------------------------------------------------------------------
# Stroke Transfer for Participating Media
# http://cg.it.aoyama.ac.jp/yonghao/sig25/abstsig25.html
#
# File: bvh.py
# Maintainer: Naoto Shirashima
#
# Description:
# Building a Bouding Volume Hierachy
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
import copy
import numpy as np
import taichi as ti
import taichi.math as tm


@ti.func
def intersect_vertex(org, dir, v0, v1, v2):
    epsilon = 1e-6
    result_ray_t = -1.0
    normal = tm.cross(v1 - v0, v2 - v0)
    normal_normalized = tm.normalize(normal)
    if tm.dot(normal_normalized, dir) >= 0:
        pass
    else:
        distance = tm.dot(normal, v0 - org) / tm.dot(normal, dir)
        collision_point = org + distance * dir
        edge0 = v1 - v0
        edge1 = v2 - v1
        edge2 = v0 - v2
        c0 = tm.cross(edge0, collision_point - v0)
        c1 = tm.cross(edge1, collision_point - v1)
        c2 = tm.cross(edge2, collision_point - v2)
        if(tm.dot(c0, c1) > 0 and tm.dot(c0, c2) > 0):
            collision_point = collision_point
            result_ray_t = distance
        else:
            collision_point = tm.vec3([tm.inf, tm.inf, tm.inf])
    return result_ray_t

def obj_to_vertex(obj):
    vertex_list = []
    # for f in obj.faces_np:
    # print('obj.faces_np', obj.faces_np)
    for index, f in enumerate(obj.faces_np):
        vertex_list.append(np.array([obj.vertices_np[f[0]], obj.vertices_np[f[1]], obj.vertices_np[f[2]]]))
    result = np.array(vertex_list)
    # print('obj_to_vertex', result)
    return result

@ti.data_oriented
class BVH:
    def __init__(self, vertex_list):
        print('vertex_list', vertex_list.shape)
        self.root = BVHNode(vertex_list, list(range(vertex_list.shape[0])), None)
        total = self.root.total
        self.bvh_obj_id = ti.field(ti.i32)
        self.bvh_left_id = ti.field(ti.i32)
        self.bvh_right_id = ti.field(ti.i32)
        self.bvh_next_id = ti.field(ti.i32)
        self.bvh_min = ti.Vector.field(3, dtype=ti.f32)
        self.bvh_max = ti.Vector.field(3, dtype=ti.f32)
        ti.root.dense(ti.i, total).place(self.bvh_obj_id, self.bvh_left_id,
                                        self.bvh_right_id, self.bvh_next_id,
                                        self.bvh_min, self.bvh_max)
        self.build()
        self.vertex0 = tm.vec3.field()
        self.vertex1 = tm.vec3.field()
        self.vertex2 = tm.vec3.field()
        ti.root.dense(ti.i, len(vertex_list)).place(self.vertex0, self.vertex1, self.vertex2)
        for i in range(len(vertex_list)):
            self.vertex0[i] = vertex_list[i][0]
            self.vertex1[i] = vertex_list[i][1]
            self.vertex2[i] = vertex_list[i][2]
        
        
    def build(self):
        i = 0
        def walk_bvh(node):
            nonlocal i
            node.id = i
            i += 1
            if node.left:
                walk_bvh(node.left)
            if node.right:
                walk_bvh(node.right)
        walk_bvh(self.root)
        def save_bvh(node):
            id = node.id
            # print('id', id)
            self.bvh_obj_id[id] = node.obj_id if node.obj is not None else -1
            self.bvh_left_id[
                id] = node.left.id if node.left is not None else -1
            self.bvh_right_id[
                id] = node.right.id if node.right is not None else -1
            self.bvh_next_id[
                id] = node.next.id if node.next is not None else -1
            self.bvh_min[id] = node.box_min
            self.bvh_max[id] = node.box_max
            if node.left is not None:
                save_bvh(node.left)
            if node.right is not None:
                save_bvh(node.right)
        save_bvh(self.root)
        self.bvh_root = 0
    
    @ti.func
    def hit_aabb(self, bvh_id, ray_origin, ray_direction, t_min, t_max):
        intersect = 1
        min_aabb = self.bvh_min[bvh_id]
        max_aabb = self.bvh_max[bvh_id]
        # print('min_aabb', min_aabb)
        # print('max_aabb', max_aabb)
        for i in ti.static(range(3)):
            if ray_direction[i] == 0:
                if ray_origin[i] < min_aabb[i] or ray_origin[i] > max_aabb[i]:
                    intersect = 0
            else:
                i1 = (min_aabb[i] - ray_origin[i]) / ray_direction[i]
                i2 = (max_aabb[i] - ray_origin[i]) / ray_direction[i]
                new_t_max = ti.max(i1, i2)
                new_t_min = ti.min(i1, i2)
                t_max = ti.min(new_t_max, t_max)
                t_min = ti.max(new_t_min, t_min)
        if t_min > t_max:
            intersect = 0
        return intersect
    
    @ti.func
    def Intersect(self, ray_origin, ray_direction):
        # print('Intersect in BVH class')
        hit_anything = False
        t_min = 0.0001
        closest_so_far = -1.0
        hit_index = -1
        i = 0
        curr = self.bvh_root
        # walk the bvh tree
        while curr != -1:
            # print('curr', curr)
            obj_id, left_id, right_id, next_id = self.get_full_id(curr)
            # print('obj_id', obj_id)
            # print('left_id', left_id)
            # print('right_id', right_id)
            # print('next_id', next_id)
            if obj_id != -1:
                # print('leaf node', obj_id)
                # leaf node
                # print('v0', self.vertex0[obj_id])
                # print('v1', self.vertex1[obj_id])
                # print('v2', self.vertex2[obj_id])
                t = intersect_vertex(ray_origin, ray_direction, self.vertex0[obj_id], self.vertex1[obj_id], self.vertex2[obj_id])
                # print('t', t)
                hit = (t < closest_so_far or hit_index == -1) and t > t_min
                # print('hit', hit)
                if hit:
                    hit_anything = True
                    closest_so_far = t
                    hit_index = obj_id
                curr = next_id
            else:
                # print('non-leaf node')
                if self.hit_aabb(curr, ray_origin, ray_direction, t_min, tm.inf if closest_so_far < 0.0 else closest_so_far):
                    # add left and right children
                    if left_id != -1:
                        curr = left_id
                    elif right_id != -1:
                        curr = right_id
                    else:
                        curr = next_id
                else:
                    curr = next_id
        # print('hit_index', hit_index)
        return closest_so_far, hit_index
    
    @ti.func
    def get_full_id(self, i):
        return self.bvh_obj_id[i], self.bvh_left_id[i], self.bvh_right_id[
            i], self.bvh_next_id[i]

def sort_obj_list(obj_list, obj_id_list):
    np.set_printoptions(threshold=np.inf)
    # print('obj_list',obj_list.shape , obj_list)
    # print('obj_list[:,:,0]', obj_list[:,:,0])
    center_x = obj_list[:,:,0].mean(axis=1)
    center_y = obj_list[:,:,1].mean(axis=1)
    center_z = obj_list[:,:,2].mean(axis=1)
    # print('center_x', center_x.shape, center_x)
    # print('center_y', center_y.shape, center_y)
    # print('center_z', center_z.shape, center_z)

    centers = np.array([center_x, center_y, center_z])
    min_center = [
        min(center_x),
        min(center_y),
        min(center_z)
    ]
    max_center = [
        max(center_x),
        max(center_y),
        max(center_z)
    ]
    span_x, span_y, span_z = (max_center[0] - min_center[0],
                                max_center[1] - min_center[1],
                                max_center[2] - min_center[2])
    if span_x >= span_y and span_x >= span_z:
        sort_index = np.argsort(center_x)
    elif span_y >= span_z:
        sort_index = np.argsort(center_y)
    else:
        sort_index = np.argsort(center_z)
    sorted_obj_list = obj_list[sort_index]
    sorted_obj_id_list = (np.array(obj_id_list)[sort_index]).tolist()
    return sorted_obj_list, sorted_obj_id_list

class BVHNode:
    left = None
    right = None
    obj = None
    obj_id = -1
    box_min = box_max = []
    id = 0
    parent = None
    total = 0
    
    def __init__(self, _obj_list ,_obj_id_list , parent=None):
        # print('BVHNode.__init__')
        obj_list = copy.copy(_obj_list)
        obj_id_list = copy.copy(_obj_id_list)
        # print('obj_list', obj_list)
        # print('obj_id_list', obj_id_list)
        self.parent = parent
        span = len(obj_list)
        # print('-----------------------')
        # print('span', span)
        if span == 1:
            self.obj = obj_list[0]
            # print('obj_list[0]', obj_list[0])
            self.box_min = obj_list[0].min(axis=0)
            self.box_max = obj_list[0].max(axis=0)
            # print('box_min', self.box_min)
            # print('box_max', self.box_max)
            self.total = 1
            self.obj_id = obj_id_list[0]
            # print('self.obj_id', self.obj_id)
        else:
            obj_list, obj_id_list_sorted = sort_obj_list(obj_list, obj_id_list=obj_id_list)
            # print('obj_list', obj_list)
            # print('obj_id_list_sorted', obj_id_list_sorted)
            obj_id_list = obj_id_list_sorted
            mid = int(span / 2)
            # print('left')
            # print('mid', mid)
            # print('obj_list[:mid]', obj_list[:mid])
            # print('obj_id_list[:mid]', obj_id_list[:mid])
            self.left = BVHNode(_obj_list=obj_list[:mid], _obj_id_list=obj_id_list[:mid], parent=self)
            # print('self.left', self.left)
            # print('right')
            self.right = BVHNode(_obj_list=obj_list[mid:], _obj_id_list=obj_id_list[mid:], parent=self)
            # print('self.right', self.right)
            self.box_min = [min(self.left.box_min[i], self.right.box_min[i]) for i in range(3)]
            self.box_max = [max(self.left.box_max[i], self.right.box_max[i]) for i in range(3)]
            self.total = self.left.total + self.right.total + 1
            # print('self.total', self.total)
        # print('-- BVHNode.__init__ end --')
            
    @property
    def next(self):
        node = self
        while True:
            if node.parent is not None and node.parent.right is not node:
                return node.parent.right
            elif node.parent is None:
                return None
            else:
                node = node.parent
        return None
