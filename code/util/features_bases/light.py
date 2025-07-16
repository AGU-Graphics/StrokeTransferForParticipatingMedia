# -----------------------------------------------------------------------------
# Stroke Transfer for Participating Media
# http://cg.it.aoyama.ac.jp/yonghao/sig25/abstsig25.html
#
# File: light.py
# Maintainer: Naoto Shirashima
#
# Description:
# Light Class
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
class Light:
    def __init__(self, position: tm.vec3, color: tm.vec3, distance_attenuation:bool = False):
        self.position = position
        self.color = color
        self.distance_attenuation = distance_attenuation
    
    # cat to print
    def __str__(self):
        return f'position:{self.position[0]},{self.position[1]},{self.position[2]}. color:{self.color[0]},{self.color[1]},{self.color[2]}. distance_attenuation:{self.distance_attenuation}'
