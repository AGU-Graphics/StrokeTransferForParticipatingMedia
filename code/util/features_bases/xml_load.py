# -----------------------------------------------------------------------------
# Stroke Transfer for Participating Media
# http://cg.it.aoyama.ac.jp/yonghao/sig25/abstsig25.html
#
# File: xml_load.py
# Maintainer: Naoto Shirashima and Yuki Yamaoka
#
# Description:
# Importing an XML file
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
import xml.etree.ElementTree as ET
from camera import Camera
from light import Light
import taichi as ti
import taichi.math as tm

def blender_axis_to_opengl_axis(blender_axis):
    return tm.vec3( [blender_axis[0], blender_axis[2], -blender_axis[1]] )

def ListToVec3(list):
    return tm.vec3([float(list[0]), float(list[1]), float(list[2])])
class xml_load:
    def __init__(self, xml_file_path):
        tree = ET.parse(xml_file_path)
        root = tree.getroot()
        self.root = root

        # camera
        camera = root.find('camera')
        camera_position = camera.attrib['position'].split(' ')
        camera_direction = camera.attrib['direction'].split(' ')
        camera_pixels = camera.attrib['resolution'].split(' ')
        camera_films = camera.attrib['film'].split(' ')
        camera_fov = camera.attrib['fov2'].split(' ')
        self.cam = Camera(
            ListToVec3(camera_position),
            ListToVec3(camera_direction),
            [int(camera_pixels[0]), int(camera_pixels[1])],
            [float(camera_films[0]), float(camera_films[1])],
            [float(camera_fov[0]), float(camera_fov[1])]
        )
        
        # light
        light_root = root.find('lights')
        lights = light_root.findall('light')
        num_lights = len(lights)
        print(f"Number of lights detected: {num_lights}\n")
        self.light = []
        for i, light in enumerate(lights, start=1):
            pos = light.get('position').split(' ')
            col = light.get('color').split(' ')
            light_distance_attenuation = False
            try:
                boolean_char = light.attrib['distance_attenuation']
                if boolean_char == 't':
                    light_distance_attenuation = True
                else:
                    light_distance_attenuation = False
            except:
                light_distance_attenuation = False
            self.light.append(
                Light(
                    position=ListToVec3(pos),
                    color=ListToVec3(col),
                    distance_attenuation=light_distance_attenuation,
                )
            )
            print('light index ', i)
            print(str(self.light[len(self.light)-1]))
            
        # attribute media
        media_root = root.find('media1')
        self.temp_min = media_root.attrib['temperature_min'].split(' ')
        self.temp_max = media_root.attrib['temperature_max'].split(' ')
        self.temp_factor = media_root.attrib['temperature_factor'].split(' ')
        self.blackbody_factor = media_root.attrib['blackbody_factor'].split(' ')

        # attribute compute
        compute_root = root.find('compute')
        self.grid_gaussian_filter_sigma = compute_root.attrib['grid_gaussian_filter_sigma'].split(' ')
        self.cell_super_samples_xml = compute_root.attrib['cell_super_samples'].split(' ')
        self.screen_super_samples_xml = compute_root.attrib['screen_super_samples'].split(' ')
        self.step_distance = compute_root.attrib['step_distance'].split(' ')
    
    
    def get_camera(self):
        return self.cam
    def get_light(self):
        return self.light
    
    # attribute media
    def get_albedo(self, media_index=1):
        media_root = self.root.find(f'media{media_index}').attrib['albedo'].split(' ')
        return ListToVec3(media_root)
    def get_albedo2(self, media_index=1):
        media1_root = self.root.find(f'media{media_index}')
        try:
            albedo2 = media1_root.attrib['albedo2'].split(' ')
        except KeyError:
            print("Available attributes in media1:", media1_root.attrib.keys())
            raise KeyError(f"Attribute 'albedo2' not found in media{media_index}.")
        return ListToVec3(albedo2)
    def get_temp_min(self):
        return float(self.temp_min[0])
    def get_temp_max(self):
        return float(self.temp_max[0])
    def get_temp_factor(self):
        return float(self.temp_factor[0])
    def get_blackbody_factor(self):
        return float(self.blackbody_factor[0])
    def get_grid_gaussian_filter_sigma(self):
        return float(self.grid_gaussian_filter_sigma[0])
    def get_phase_function_HG(self, media_index=1):
        g = self.root.find(f'media{media_index}').attrib['phase_function_g1'].split(' ')
        return float(g[0])
    def get_phase_function_HG2(self, media_index=1):
        g = self.root.find(f'media{media_index}').attrib['phase_function_g2'].split(' ')
        return float(g[0])
    def get_phase_function_HG_blend(self, media_index=1):
        g_blend = self.root.find(f'media{media_index}').attrib['phase_function_g_blend'].split(' ')
        return float(g_blend[0])

    
    # attribute compute
    def get_step_distance(self):
        return float(self.step_distance[0])
    def get_screen_super_samples(self):
        return int(self.screen_super_samples_xml[0])
    def get_cell_super_samples(self):
        return int(self.cell_super_samples_xml[0])