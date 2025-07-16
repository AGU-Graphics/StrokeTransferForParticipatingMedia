# -----------------------------------------------------------------------------
# Stroke Transfer for Participating Media
# http://cg.it.aoyama.ac.jp/yonghao/sig25/abstsig25.html
#
# File: obj_loader.py
# Maintainer: Naoto Shirashima
#
# Description:
# Loading .obj files
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
from data_io import DataIO3D

class ObjLoader:
    def __init__(self, filePath):
        numVertices = 0
        numUVs = 0
        numNormals = 0
        numFaces = 0

        vertices = []
        uvs = []
        normals = []
        vertexColors = []
        faceVertIDs = []
        uvIDs = []
        normalIDs = []

        for line in open(filePath, "r"):
            vals = line.split()

            if len(vals) == 0:
                continue

            if vals[0] == "v":
                v = np.array([float(vals[1]), float(vals[2]), float(vals[3])])
                vertices.append(v)

                if len(vals) == 7:
                    vc = map(float, vals[4:7])
                    vertexColors.append(vc)

                numVertices += 1
            if vals[0] == "vt":
                vt = map(float, vals[1:3])
                # vt = np.array([float(vals[1]), float(vals[2])])
                uvs.append(vt)
                numUVs += 1
            if vals[0] == "vn":
                vn = map(float, vals[1:4])
                # vn = np.array([float(vals[1]), float(vals[2]), float(vals[3])])
                normals.append(vn)
                numNormals += 1
            if vals[0] == "f":
                fvID = []
                uvID = []
                nvID = []
                for f in vals[1:]:
                    w = f.split("/")

                    if numVertices > 0:
                        fvID.append(int(w[0])-1)

                    if numUVs > 0:
                        uvID.append(int(w[1])-1)

                    if numNormals > 0:
                        nvID.append(int(w[2])-1)

                faceVertIDs.append(fvID)
                uvIDs.append(uvID)
                normalIDs.append(nvID)

                numFaces += 1

        # print ("numVertices: ", numVertices)
        # print ("numUVs: ", numUVs)
        # print ("numNormals: ", numNormals)
        # print ("numFaces: ", numFaces)

        self.vertices = ti.field(tm.vec3, numVertices)
        self.faces = ti.field(tm.vec3, numFaces)

        self.vertices.from_numpy(np.array(vertices).astype(np.float32))
        self.vertices_np = vertices
        self.faces.from_numpy(np.array(faceVertIDs).astype(np.int32))
        self.faces_np = faceVertIDs


        # test
        self.center = ti.field(tm.vec3, shape=(self.faces.shape[0]))

    @ti.func
    def get_vertices(self, vertex_index):
        v0 = self.vertices[int(vertex_index[0])]
        v1 = self.vertices[int(vertex_index[1])]
        v2 = self.vertices[int(vertex_index[2])]

        return v0, v1, v2

    def ComputeCenter(self):
        @ti.kernel
        def ComputeCenterKernel():
            for i in ti.grouped(self.faces):
                v0, v1, v2 = self.get_vertices(self.faces[i])
                self.center[i] = tm.vec3((v0 + v1 + v2) / 3)
        ComputeCenterKernel()

    @ti.func
    def IntersectFace(self, ray_origin, ray_direction, vertices) -> tm.vec3:
        epsilon = 1e-6
        result_ray_t = tm.inf
        v0, v1, v2 = self.get_vertices(vertices)
        normal = tm.cross(v1 - v0, v2 - v0)
        collision_point = tm.vec3([tm.inf, tm.inf, tm.inf])
        if 1 - tm.dot(normal, ray_direction) <= epsilon or ti.abs(tm.dot(normal, ray_direction)) <= epsilon:
            collision_point = tm.vec3([tm.inf, tm.inf, tm.inf])
        else:
            distance = tm.dot(normal, v0 - ray_origin) / tm.dot(normal, ray_direction)
            collision_point = ray_origin + distance * ray_direction
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


    @ti.func
    def IntersectOBJ(self, ray_origin, ray_direction) -> tm.vec3:
        ray_distance = tm.inf
        intersect_face_index = -1
        for i in range(self.faces.shape[0]):
            tmp_ray_distance = self.IntersectFace(ray_origin, ray_direction, self.faces[i])
            if tmp_ray_distance > 1e-3 and tmp_ray_distance < ray_distance:
                ray_distance = tmp_ray_distance
                intersect_face_index = i
        return ray_distance,intersect_face_index

