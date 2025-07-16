# -----------------------------------------------------------------------------
# Stroke Transfer for Participating Media
# http://cg.it.aoyama.ac.jp/yonghao/sig25/abstsig25.html
#
# File: p2_compute_features.py
# Maintainer: Naoto Shirashima
#
# Description:
# Feature and basis field calculation
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
import subprocess
import xml.etree.ElementTree as ET
import os
import sys
import json
import shutil
from subprocess import SubprocessError
import asyncio
from tqdm import tqdm
import time
import traceback
import numpy as np
_file_dir = os.path.dirname( os.path.abspath( __file__ ) )
_util_dir = os.path.abspath( os.path.join( _file_dir, "util/features_bases" ) )
sys.path.append( _util_dir )

def frame_skips_func( frame_skip ):
    return np.array( [ 96, 48, 24, 12, 6, 4, 2, 1 ] ) * frame_skip

def unique_list( frame_start, frame_end, frame_skip ):
    target_frames = np.array( [], dtype=int )
    skip_list = frame_skips_func( frame_skip )
    for fs in skip_list:
        target_frames = np.append( target_frames, np.arange( frame_start, frame_end+1, fs ) )
    indexes = np.unique( target_frames, return_index=True )[1]
    target_frames = [ target_frames[ index ] for index in sorted( indexes ) ]
    return target_frames

class ProcessExecutionError(Exception):
    def __init__(self, returncode, cmd, stdout, stderr):
        super().__init__(returncode, cmd, stdout, stderr)
        self.returncode = returncode
        self.cmd = cmd
        self.stdout = stdout
        self.stderr = stderr

async def run( run_command, release_mode=True, shell=False ):
    if isinstance(run_command, list):
        # run_command = ' '.join(run_command)
        tmp_command = ''
        for cmd in run_command:
            if cmd == '':
                tmp_command += '"" '
            else:
                tmp_command += cmd + ' '
        run_command = tmp_command
    proc = await asyncio.create_subprocess_shell(
        run_command,
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.PIPE,
    )
    stdout, stderr = await proc.communicate()
    if proc.returncode == 0:
        #print(f'Command failed: {run_command}')
        #print('-'*10, 'STDOUT', '-'*10)
        #print(stdout.decode())
        pass
    if proc.returncode != 0:
        print(f'Command failed: {run_command}')
        print('-'*10, 'STDOUT', '-'*10)
        print(stdout.decode())
        print('-'*10, 'STDERR', '-'*10)
        print(stderr.decode())
        print('-'*10, 'END', '-'*10)
        raise ProcessExecutionError(
            proc.returncode,
            run_command,
            stdout.decode(),
            stderr.decode()
        )

def list_to_vector_string( list ):
    return f"{list[0]} {list[1]} {list[2]}"

def blender_axes_to_opengl_axis( blender_axes ):
    return [ blender_axes[0], blender_axes[2], -blender_axes[1] ]
    
def prepare_internal_dirs( scene_name, settings ):
    internals = settings[ "internal_file_templates" ]
    for i in internals:
        if isinstance( settings[ "internal_file_templates" ][ i ], list ):
            for j in settings[ "internal_file_templates" ][ i ]:
                fn = "./" + scene_name + "/" + j
                fn_dir = fn[ :fn.rfind( "/" ) ]
                os.makedirs( fn_dir, exist_ok = True )
        else:
            fn = "./" + scene_name + "/" + settings[ "internal_file_templates" ][ i ]
            fn_dir = fn[ :fn.rfind( "/" ) ]
            os.makedirs( fn_dir, exist_ok = True )
    
    internals = settings[ "raw_feature_file_templates" ]
    for i in internals:
        if settings[ "raw_feature_file_templates" ][ i ] is None:
            continue
        fn = "./" + scene_name + "/" + settings[ "raw_feature_file_templates" ][ i ]
        fn_dir = fn[ :fn.rfind( "/" ) ]
        os.makedirs( fn_dir, exist_ok = True )
    
    internals = settings[ "basis_file_templates" ]
    for i in internals:
        if settings[ "basis_file_templates" ][ i ] is None:
            continue
        fn = "./" + scene_name + "/" + settings[ "basis_file_templates" ][ i ]
        fn_dir = fn[ :fn.rfind( "/" ) ]
        os.makedirs( fn_dir, exist_ok = True )

def remove_internal_dirs( scene_name, settings ):
    internals = settings[ "internal_file_templates" ]
    
    for i in internals:
        fn = "./" + scene_name + "/" + settings[ "internal_file_templates" ][ i ]
        fn_dir = fn[ :fn.rfind( "/" ) ]
        shutil.rmtree( fn_dir )

def remove_tmp_frame( scene_name, settings, target_frame ):
    internals = settings[ "internal_file_templates" ]

    for i in internals:
        if settings[ "internal_file_templates" ][ i ] is None:
            continue
        fn = "./" + scene_name + "/" + settings[ "internal_file_templates" ][ i ] % target_frame
        if os.path.exists( fn ):
            os.remove( fn )

def template_path_list( dir, settings, attribute_names, frame_index ):
    while True:
        settings = settings[attribute_names[0]]
        if len(attribute_names) == 1:
            break
        attribute_names = attribute_names[1:]
    
    result = []
    for i in settings:
        if i[-4:] == 'None':
            result.append(None)
        else:
            if '%' in i:
                result.append( os.path.join(dir, i % frame_index) )
            else:
                result.append(os.path.join(dir, i))
    if len(result) == 0:
        result = None    
    return result

def template_path( template_path, frame_index ):
    
    if template_path[-4:] == 'None':
        return None
    else:
        if '%' in template_path:
            return template_path % frame_index
        else:
            return template_path

async def compute_f_cs( scene_name, settings, target_frame, xml_path1, xml_path2 ):
    silhouette_distance_threshold_value = 0.9
    python_command = settings[ 'python_cmd' ]
    features_bases_dir = '../code/util/features_bases'
    fps = settings[ "frame_settings" ][ "fps" ]
    special_scene_type = settings.get('special_scene_type', None)
    
    special_scene_type_cloud = 'clouds'
    special_scene_type_mix_media = 'mix_media'
    special_scene_type_surface = 'surface'
    if special_scene_type is not None and special_scene_type not in [special_scene_type_cloud, special_scene_type_mix_media, special_scene_type_surface]:
        raise ValueError(f"Invalid special_scene_type: {special_scene_type}.")

    original_density_path = template_path(f"./{scene_name}/assets/volume_data/{settings['volume_data']['density']}", target_frame)
    original_velocity_x_path = template_path(f"./{scene_name}/assets/volume_data/{settings['volume_data']['velocity_x']}", target_frame)
    original_velocity_y_path = template_path(f"./{scene_name}/assets/volume_data/{settings['volume_data']['velocity_y']}", target_frame)
    original_velocity_z_path = template_path(f"./{scene_name}/assets/volume_data/{settings['volume_data']['velocity_z']}", target_frame)
    original_temperature_path = template_path(f"./{scene_name}/assets/volume_data/{settings['volume_data']['temperature']}", target_frame)
    if original_temperature_path is not None:
        background_temperature = ET.parse(xml_path1).getroot().find('media1').attrib['background_temperature']
        surface_temperature = ET.parse(xml_path1).getroot().find('media1').attrib['surface_temperature']
        media_temperature_min = ET.parse(xml_path1).getroot().find('media1').attrib['temperature_min']
        media_temperature_max = ET.parse(xml_path1).getroot().find('media1').attrib['temperature_max']
    
    density_sigma = ET.parse(xml_path1).getroot().find('media1').attrib['sigma_k']
    grid_gaussian_filter_sigma = ET.parse(xml_path1).getroot().find(f'compute').attrib['grid_gaussian_filter_sigma']
    
    
    
    # internal files
    extinction_path = template_path(f"./{scene_name}/{settings['internal_file_templates']['extinction']}", target_frame)
    integrated_extinction_cell_path = template_path(f"./{scene_name}/{settings['internal_file_templates']['integrated_extinction_cell']}", target_frame)
    transmittance_cell_path = template_path(f"./{scene_name}/{settings['internal_file_templates']['transmittance_cell']}", target_frame)
    transmittance_cell_gaussian_filtered_path = template_path(f"./{scene_name}/{settings['internal_file_templates']['transmittance_cell_gaussian_filtered']}", target_frame)
    fpd_path = template_path(f"./{scene_name}/{settings['internal_file_templates']['fpd']}", target_frame)
    screen_integrated_extinction_path = template_path(f"./{scene_name}/{settings['internal_file_templates']['screen_integrated_extinction']}", target_frame)
    screen_integrated_extinction_extra_path = template_path(f"./{scene_name}/{settings['internal_file_templates']['screen_integrated_extinction_extra']}", target_frame)
    transmittance_screen_extra_path = template_path(f"./{scene_name}/{settings['internal_file_templates']['transmittance_screen_extra']}", target_frame)
    silhouette_distance_gradient_path = template_path(f"./{scene_name}/{settings['internal_file_templates']['silhouette_distance_gradient']}", target_frame)
    luminance_cell_r_path = template_path(f"./{scene_name}/{settings['internal_file_templates']['luminance_cell_r']}", target_frame)
    luminance_cell_g_path = template_path(f"./{scene_name}/{settings['internal_file_templates']['luminance_cell_g']}", target_frame)
    luminance_cell_b_path = template_path(f"./{scene_name}/{settings['internal_file_templates']['luminance_cell_b']}", target_frame)
    luminance_r_fpd_path = template_path(f"./{scene_name}/{settings['internal_file_templates']['luminance_r_fpd']}", target_frame)
    luminance_g_fpd_path = template_path(f"./{scene_name}/{settings['internal_file_templates']['luminance_g_fpd']}", target_frame)
    luminance_b_fpd_path = template_path(f"./{scene_name}/{settings['internal_file_templates']['luminance_b_fpd']}", target_frame)
    rel_velocity_screen_space_u_path = template_path(f"./{scene_name}/{settings['internal_file_templates']['rel_velocity_screen_space_u']}", target_frame)
    rel_velocity_screen_space_v_path = template_path(f"./{scene_name}/{settings['internal_file_templates']['rel_velocity_screen_space_v']}", target_frame)
    rel_velocity_screen_space_u_fpd_path = template_path(f"./{scene_name}/{settings['internal_file_templates']['rel_velocity_screen_space_u_fpd']}", target_frame)
    rel_velocity_screen_space_v_fpd_path = template_path(f"./{scene_name}/{settings['internal_file_templates']['rel_velocity_screen_space_v_fpd']}", target_frame)
    world_scape_apparent_normal_path = template_path(f"./{scene_name}/{settings['internal_file_templates']['world_scape_apparent_normal']}", target_frame)
    # view_space_apparent_normal_path = template_path(f"./{scene_name}/{settings['internal_file_templates']['view_space_apparent_normal']}", target_frame)
    luminance_l_gradient_path = template_path(f"./{scene_name}/{settings['internal_file_templates']['luminance_l_gradient']}", target_frame)
    mean_free_path_cell_path = template_path(f"./{scene_name}/{settings['internal_file_templates']['mean_free_path_cell']}", target_frame)
    mean_free_path_cell_fpd_path = template_path(f"./{scene_name}/{settings['internal_file_templates']['mean_free_path_cell_fpd']}", target_frame)
    mean_free_path_gradient_path = template_path(f"./{scene_name}/{settings['internal_file_templates']['mean_free_path_gradient']}", target_frame)
    if original_temperature_path is not None:
        temperature_surface_and_background_path = template_path(f"./{scene_name}/{settings['internal_file_templates']['temperature_surface_and_background']}", target_frame)
        temperature_surface_and_background_transmittance_path = template_path(f"./{scene_name}/{settings['internal_file_templates']['temperature_surface_and_background_transmittance']}", target_frame)
        temperature_fpd_path = template_path(f"./{scene_name}/{settings['internal_file_templates']['temperature_fpd']}", target_frame)
        temperature_raw_media_path = template_path(f"./{scene_name}/{settings['internal_file_templates']['temperature_raw_media']}", target_frame)
        temperature_media_path = template_path(f"./{scene_name}/{settings['internal_file_templates']['temperature_media']}", target_frame)
    if special_scene_type == special_scene_type_cloud:
        transmittance_cell_inside_path = template_path(f"./{scene_name}/{settings['internal_file_templates']['transmittance_cell_inside']}", target_frame)
        integrated_extinction_cell_inside_path = template_path(f"./{scene_name}/{settings['internal_file_templates']['integrated_extinction_cell_inside']}", target_frame)
        fpd_inside_path = template_path(f"./{scene_name}/{settings['internal_file_templates']['fpd_inside']}", target_frame)
        extinction_whole_path = "./" + scene_name + "/" + settings[ "internal_file_templates" ][ "extinction_whole" ] % target_frame
        extinction_inside_path = extinction_path
        extinction_outside_path = "./" + scene_name + "/" + settings[ "internal_file_templates" ][ "extinction_outside" ] % target_frame
        # fpd_whole_path = template_path(f"./{scene_name}/{settings['internal_file_templates']['fpd_whole']}", target_frame)
    if special_scene_type == special_scene_type_surface:
        depth_value_path = template_path(f"./{scene_name}/{settings['internal_file_templates']['depth_value']}", target_frame)
        depth_value_extra_path = template_path(f"./{scene_name}/{settings['internal_file_templates']['depth_value_extra']}", target_frame)
        transmittance_screen_media_path = template_path(f"./{scene_name}/{settings['internal_file_templates']['transmittance_screen_media']}", target_frame)
        surface_transmittance_path = template_path(f"./{scene_name}/{settings['internal_file_templates']['surface_transmittance']}", target_frame)
        transmittance_screen_extra_media_path = template_path(f"./{scene_name}/{settings['internal_file_templates']['transmittance_screen_extra_media']}", target_frame)
        surface_transmittance_extra_path = template_path(f"./{scene_name}/{settings['internal_file_templates']['surface_transmittance_extra']}", target_frame)
        shadow_test_path = template_path_list(f"./{scene_name}", settings, ['internal_file_templates', 'shadow_test'], target_frame)
        luminance_cell_r_shadow_test_path = template_path_list(f"./{scene_name}", settings, ['internal_file_templates', 'luminance_cell_r_shadow_test'], target_frame)
        luminance_cell_g_shadow_test_path = template_path_list(f"./{scene_name}", settings, ['internal_file_templates', 'luminance_cell_g_shadow_test'], target_frame)
        luminance_cell_b_shadow_test_path = template_path_list(f"./{scene_name}", settings, ['internal_file_templates', 'luminance_cell_b_shadow_test'], target_frame)
        luminance_cell_r_shadow_accumulated_path = template_path(f"./{scene_name}/{settings['internal_file_templates']['luminance_cell_r_shadow_accumulated']}", target_frame)
        luminance_cell_g_shadow_accumulated_path = template_path(f"./{scene_name}/{settings['internal_file_templates']['luminance_cell_g_shadow_accumulated']}", target_frame)
        luminance_cell_b_shadow_accumulated_path = template_path(f"./{scene_name}/{settings['internal_file_templates']['luminance_cell_b_shadow_accumulated']}", target_frame)
        intensity_r_media_path = template_path(f"./{scene_name}/{settings['internal_file_templates']['intensity_r_media']}", target_frame)
        intensity_g_media_path = template_path(f"./{scene_name}/{settings['internal_file_templates']['intensity_g_media']}", target_frame)
        intensity_b_media_path = template_path(f"./{scene_name}/{settings['internal_file_templates']['intensity_b_media']}", target_frame)
        integrated_extinction_cell_from_light_path = template_path_list(f"./{scene_name}", settings, ['internal_file_templates', 'integrated_extinction_cell_from_light'], target_frame)
        transmittance_cell_from_light_path = template_path_list(f"./{scene_name}", settings, ['internal_file_templates', 'transmittance_cell_from_light'], target_frame)
        transmittance_cell_from_light_accumulated_path = template_path(f"./{scene_name}/{settings['internal_file_templates']['transmittance_cell_from_light_accumulated']}", target_frame)
        intensity_r_surface_path = template_path(f"./{scene_name}/{settings['internal_file_templates']['intensity_r_surface']}", target_frame)
        intensity_g_surface_path = template_path(f"./{scene_name}/{settings['internal_file_templates']['intensity_g_surface']}", target_frame)
        intensity_b_surface_path = template_path(f"./{scene_name}/{settings['internal_file_templates']['intensity_b_surface']}", target_frame)
        intensity_r_transmittance_path = template_path(f"./{scene_name}/{settings['internal_file_templates']['intensity_r_transmittance']}", target_frame)
        intensity_g_transmittance_path = template_path(f"./{scene_name}/{settings['internal_file_templates']['intensity_g_transmittance']}", target_frame)
        intensity_b_transmittance_path = template_path(f"./{scene_name}/{settings['internal_file_templates']['intensity_b_transmittance']}", target_frame)
        apparent_relative_velocity_u_media_path = template_path(f"./{scene_name}/{settings['internal_file_templates']['apparent_relative_velocity_u_media']}", target_frame)
        apparent_relative_velocity_v_media_path = template_path(f"./{scene_name}/{settings['internal_file_templates']['apparent_relative_velocity_v_media']}", target_frame)
        apparent_relative_velocity_u_surface_path = template_path(f"./{scene_name}/{settings['internal_file_templates']['apparent_relative_velocity_u_surface']}", target_frame)
        apparent_relative_velocity_v_surface_path = template_path(f"./{scene_name}/{settings['internal_file_templates']['apparent_relative_velocity_v_surface']}", target_frame)
        apparent_relative_velocity_u_surface_transmittance_path = template_path(f"./{scene_name}/{settings['internal_file_templates']['apparent_relative_velocity_u_surface_transmittance']}", target_frame)
        apparent_relative_velocity_v_surface_transmittance_path = template_path(f"./{scene_name}/{settings['internal_file_templates']['apparent_relative_velocity_v_surface_transmittance']}", target_frame)
        apparent_normal_x_media_path = template_path(f"./{scene_name}/{settings['internal_file_templates']['apparent_normal_x_media']}", target_frame)
        apparent_normal_y_media_path = template_path(f"./{scene_name}/{settings['internal_file_templates']['apparent_normal_y_media']}", target_frame)
        apparent_normal_z_media_path = template_path(f"./{scene_name}/{settings['internal_file_templates']['apparent_normal_z_media']}", target_frame)
        apparent_normal_x_surface_path = template_path(f"./{scene_name}/{settings['internal_file_templates']['apparent_normal_x_surface']}", target_frame)
        apparent_normal_y_surface_path = template_path(f"./{scene_name}/{settings['internal_file_templates']['apparent_normal_y_surface']}", target_frame)
        apparent_normal_z_surface_path = template_path(f"./{scene_name}/{settings['internal_file_templates']['apparent_normal_z_surface']}", target_frame)
        apparent_normal_x_surface_transmittance_path = template_path(f"./{scene_name}/{settings['internal_file_templates']['apparent_normal_x_surface_transmittance']}", target_frame)
        apparent_normal_y_surface_transmittance_path = template_path(f"./{scene_name}/{settings['internal_file_templates']['apparent_normal_y_surface_transmittance']}", target_frame)
        apparent_normal_z_surface_transmittance_path = template_path(f"./{scene_name}/{settings['internal_file_templates']['apparent_normal_z_surface_transmittance']}", target_frame)
        gaussian_curvature_media_path = template_path(f"./{scene_name}/{settings['internal_file_templates']['gaussian_curvature_media']}", target_frame)
        mean_curvature_media_path = template_path(f"./{scene_name}/{settings['internal_file_templates']['mean_curvature_media']}", target_frame)
        gaussian_curvature_surface_path = template_path(f"./{scene_name}/{settings['internal_file_templates']['gaussian_curvature_surface']}", target_frame)
        mean_curvature_surface_path = template_path(f"./{scene_name}/{settings['internal_file_templates']['mean_curvature_surface']}", target_frame)
        gaussian_curvature_surface_transmittance_path = template_path(f"./{scene_name}/{settings['internal_file_templates']['gaussian_curvature_surface_transmittance']}", target_frame)
        mean_curvature_surface_transmittance_path = template_path(f"./{scene_name}/{settings['internal_file_templates']['mean_curvature_surface_transmittance']}", target_frame)
    else:
        depth_value_path = ""
        depth_value_extra_path = ""

    # raw feature files
    intensity_r_path = template_path(f"./{scene_name}/{settings['raw_feature_file_templates']['intensity_r']}", target_frame)
    intensity_g_path = template_path(f"./{scene_name}/{settings['raw_feature_file_templates']['intensity_g']}", target_frame)
    intensity_b_path = template_path(f"./{scene_name}/{settings['raw_feature_file_templates']['intensity_b']}", target_frame)
    intensity_l_path = template_path(f"./{scene_name}/{settings['raw_feature_file_templates']['intensity_l']}", target_frame)
    intensity_l_gradient_path = template_path(f"./{scene_name}/{settings['raw_feature_file_templates']['intensity_l_gradient']}", target_frame)
    gaussian_curvature_path = template_path(f"./{scene_name}/{settings['raw_feature_file_templates']['gaussian_curvature']}", target_frame)
    mean_curvature_path = template_path(f"./{scene_name}/{settings['raw_feature_file_templates']['mean_curvature']}", target_frame)
    apparent_normal_x_path = template_path(f"./{scene_name}/{settings['raw_feature_file_templates']['apparent_normal_x']}", target_frame)
    apparent_normal_y_path = template_path(f"./{scene_name}/{settings['raw_feature_file_templates']['apparent_normal_y']}", target_frame)
    apparent_normal_z_path = template_path(f"./{scene_name}/{settings['raw_feature_file_templates']['apparent_normal_z']}", target_frame)
    apparent_relative_velocity_u_path = template_path(f"./{scene_name}/{settings['raw_feature_file_templates']['apparent_relative_velocity_u']}", target_frame)
    apparent_relative_velocity_v_path = template_path(f"./{scene_name}/{settings['raw_feature_file_templates']['apparent_relative_velocity_v']}", target_frame)
    apparent_relative_velocity_norm_path = template_path(f"./{scene_name}/{settings['raw_feature_file_templates']['apparent_relative_velocity_norm']}", target_frame)
    transmittance_screen_path = template_path(f"./{scene_name}/{settings['raw_feature_file_templates']['transmittance']}", target_frame)
    mean_free_path_path = template_path(f"./{scene_name}/{settings['raw_feature_file_templates']['mean_free_path']}", target_frame)
    silhouette_distance_path = template_path(f"./{scene_name}/{settings['raw_feature_file_templates']['silhouette_distance']}", target_frame)
    temperature_path = template_path(f"./{scene_name}/{settings['raw_feature_file_templates']['temperature']}", target_frame)
    out_bb_path = template_path(f"./{scene_name}/{settings['raw_feature_file_templates']['bounding_box']}", target_frame)
    
    # basis files
    intensity_gradient_para_path = template_path(f"./{scene_name}/{settings['basis_file_templates']['intensity_gradient_para']}", target_frame)
    intensity_gradient_perp_path = template_path(f"./{scene_name}/{settings['basis_file_templates']['intensity_gradient_perp']}", target_frame)
    silhouette_guided_para_path = template_path(f"./{scene_name}/{settings['basis_file_templates']['silhouette_guided_para']}", target_frame)
    silhouette_guided_perp_path = template_path(f"./{scene_name}/{settings['basis_file_templates']['silhouette_guided_perp']}", target_frame)
    apparent_normal_para_path = template_path(f"./{scene_name}/{settings['basis_file_templates']['apparent_normal_para']}", target_frame)
    apparent_normal_perp_path = template_path(f"./{scene_name}/{settings['basis_file_templates']['apparent_normal_perp']}", target_frame)
    apparent_relative_velocity_para_path = template_path(f"./{scene_name}/{settings['basis_file_templates']['apparent_relative_velocity_para']}", target_frame)
    apparent_relative_velocity_perp_path = template_path(f"./{scene_name}/{settings['basis_file_templates']['apparent_relative_velocity_perp']}", target_frame)
    mean_free_path_gradient_para_path = template_path(f"./{scene_name}/{settings['basis_file_templates']['mean_free_path_gradient_para']}", target_frame)
    mean_free_path_gradient_perp_path = template_path(f"./{scene_name}/{settings['basis_file_templates']['mean_free_path_gradient_perp']}", target_frame)

    compute_mode = 'gpu'
    
    # create extinction
    if special_scene_type == special_scene_type_cloud:
        # await run( f'h5dump -d "resolution" {original_density_path} && h5dump -d "resolution" {original_velocity_x_path} && stop' )
        await run( [ python_command, f'{features_bases_dir}/multiply_3d_data.py', original_density_path, density_sigma, extinction_inside_path ] )
        # await run( f'h5dump -d "resolution" {extinction_inside_path} && stop' )
        await run(f'{python_command} {features_bases_dir}/generate_sky_density_from_cloud.py {compute_mode} {extinction_path} {xml_path1} {extinction_whole_path} {extinction_outside_path}' )
        extinction_in_transmittance = extinction_whole_path
    else:
        await run( [ python_command, f'{features_bases_dir}/multiply_3d_data.py', original_density_path, density_sigma, extinction_path ] )
        extinction_in_transmittance = extinction_path
    await run( [ python_command, f'{features_bases_dir}/out_bb.py', extinction_path, out_bb_path ] )

    # transmittance cell
    await run( [ python_command, f'{features_bases_dir}/integral_cell.py', compute_mode, extinction_in_transmittance, 'camera', xml_path1, integrated_extinction_cell_path ])
    await run( f'{python_command} {features_bases_dir}/convert_to_transmittance_cell.py {integrated_extinction_cell_path} {transmittance_cell_path}'  )
    if special_scene_type == special_scene_type_cloud:
        await run(f'{python_command} {features_bases_dir}/integral_cell.py {compute_mode} {extinction_inside_path} camera {xml_path1} {integrated_extinction_cell_inside_path}')
        await run(f'{python_command} {features_bases_dir}/convert_to_transmittance_cell.py {integrated_extinction_cell_inside_path} {transmittance_cell_inside_path}' )
    # transmittance screen
    await run([python_command, f'{features_bases_dir}/line_of_sight_integration.py', compute_mode, extinction_in_transmittance, depth_value_path,'camera' , xml_path1, screen_integrated_extinction_path, '--extra_resolution', '2.0', '--extra_depth_test', depth_value_extra_path, '--output_extra_resolution', screen_integrated_extinction_extra_path])
    if special_scene_type == special_scene_type_surface:
        await run( f'{python_command} {features_bases_dir}/convert_to_transmittance_screen.py {screen_integrated_extinction_path} {transmittance_screen_media_path}' )
        await run( f'{python_command} {features_bases_dir}/convert_to_transmittance_screen.py {screen_integrated_extinction_extra_path} {transmittance_screen_extra_media_path}' )
        await run(f'h5dump -d "resolution" {transmittance_screen_extra_media_path} && h5dump -d "resolution" {surface_transmittance_extra_path}')
        await run( f'{python_command} {features_bases_dir}/blend_transmittance.py {transmittance_screen_media_path} {surface_transmittance_path} {transmittance_screen_path}' )
        # for surface transmittance extra
        # await run( f'{python_command} {features_bases_dir}/blend_transmittance.py {transmittance_screen_extra_media_path} {surface_transmittance_extra_path} {transmittance_screen_extra_path}' )
    else:
        await run( f'{python_command} {features_bases_dir}/convert_to_transmittance_screen.py {screen_integrated_extinction_path} {transmittance_screen_path}' )
        await run( f'{python_command} {features_bases_dir}/convert_to_transmittance_screen.py {screen_integrated_extinction_extra_path} {transmittance_screen_extra_path}' )
    # silhouette
    await run( f'{python_command} {features_bases_dir}/silhouette_distance.py -i {transmittance_screen_extra_path} -t {silhouette_distance_threshold_value} --extra_resolution 2.0 -o {silhouette_distance_path}' )
    await run( f'{python_command} {features_bases_dir}/gradient_feature.py {compute_mode} {silhouette_distance_path} {silhouette_distance_gradient_path}' )
    await run( f'{python_command} {features_bases_dir}/normalize_and_rotate_basis.py -i {silhouette_distance_gradient_path} -compute_mode {compute_mode} -out_perp {silhouette_guided_perp_path} -out_para {silhouette_guided_para_path}' )

    # fpd
    await run( f'{python_command} {features_bases_dir}/multiply_3d_data.py {transmittance_cell_path} {extinction_in_transmittance} {fpd_path}' )
    if special_scene_type == special_scene_type_cloud:
        await run(f'{python_command} {features_bases_dir}/multiply_3d_data.py {transmittance_cell_inside_path} {extinction_inside_path} {fpd_inside_path}')
        # await run( f'h5dump -d "resolution" {fpd_inside_path} && stop' )

    # luminance feature
    intensity_temperature_volume_path = original_temperature_path if original_temperature_path is not None else ''
    if special_scene_type == special_scene_type_mix_media:
        await run(f'{python_command} {features_bases_dir}/luminance_cell_mix_media.py -compute_mode {compute_mode} -extinction {extinction_path} -temperature {intensity_temperature_volume_path} -xml {xml_path1} -out_r {luminance_cell_r_path} -out_g {luminance_cell_g_path} -out_b {luminance_cell_b_path}', shell=True)
    elif special_scene_type == special_scene_type_cloud:
        await run( f'{python_command} {features_bases_dir}/luminance_cell_clouds.py {compute_mode} {extinction_whole_path} {extinction_inside_path} {extinction_outside_path} {xml_path1} {luminance_cell_r_path} {luminance_cell_g_path} {luminance_cell_b_path}' )
    else:
        await run(f'{python_command} {features_bases_dir}/luminance_cell.py -compute_mode {compute_mode} -extinction {extinction_path} -temperature {intensity_temperature_volume_path} -xml {xml_path1} -out_r {luminance_cell_r_path} -out_g {luminance_cell_g_path} -out_b {luminance_cell_b_path}', shell=True)
    
    if special_scene_type != special_scene_type_surface:
        await run([python_command, f'{features_bases_dir}/multiply_3d_data.py', luminance_cell_r_path, fpd_path,  luminance_r_fpd_path])
        await run([python_command, f'{features_bases_dir}/multiply_3d_data.py', luminance_cell_g_path, fpd_path,  luminance_g_fpd_path])
        await run([python_command, f'{features_bases_dir}/multiply_3d_data.py', luminance_cell_b_path, fpd_path,  luminance_b_fpd_path])
        await run([python_command, f'{features_bases_dir}/line_of_sight_integration.py', compute_mode, luminance_r_fpd_path, '', 'camera', xml_path1, intensity_r_path])
        await run([ python_command, f'{features_bases_dir}/line_of_sight_integration.py', compute_mode, luminance_g_fpd_path, '', 'camera', xml_path1, intensity_g_path ])
        await run([ python_command, f'{features_bases_dir}/line_of_sight_integration.py', compute_mode, luminance_b_fpd_path, '', 'camera', xml_path1, intensity_b_path ])
    else:
        for shadow_test_index in range(len(shadow_test_path)):
            await run( f'{python_command} {features_bases_dir}/multiply_3d_data.py {luminance_cell_r_path} {shadow_test_path[shadow_test_index]} {luminance_cell_r_shadow_test_path[shadow_test_index]}' )
            await run( f'{python_command} {features_bases_dir}/multiply_3d_data.py {luminance_cell_g_path} {shadow_test_path[shadow_test_index]} {luminance_cell_g_shadow_test_path[shadow_test_index]}' )
            await run( f'{python_command} {features_bases_dir}/multiply_3d_data.py {luminance_cell_b_path} {shadow_test_path[shadow_test_index]} {luminance_cell_b_shadow_test_path[shadow_test_index]}' )
        for shadow_test_index in range(len(shadow_test_path)-1):
            if shadow_test_index == 0:
                tmp_luminance_cell_r_shadow_test_path = luminance_cell_r_shadow_test_path[shadow_test_index]
                tmp_luminance_cell_g_shadow_test_path = luminance_cell_g_shadow_test_path[shadow_test_index]
                tmp_luminance_cell_b_shadow_test_path = luminance_cell_b_shadow_test_path[shadow_test_index]
            else:
                tmp_luminance_cell_r_shadow_test_path = luminance_cell_r_shadow_accumulated_path
                tmp_luminance_cell_g_shadow_test_path = luminance_cell_g_shadow_accumulated_path
                tmp_luminance_cell_b_shadow_test_path = luminance_cell_b_shadow_accumulated_path
            await run( f'{python_command} {features_bases_dir}/plus_3d_data.py {tmp_luminance_cell_r_shadow_test_path} {luminance_cell_r_shadow_test_path[shadow_test_index+1]} {luminance_cell_r_shadow_accumulated_path}' )
            await run( f'{python_command} {features_bases_dir}/plus_3d_data.py {tmp_luminance_cell_g_shadow_test_path} {luminance_cell_g_shadow_test_path[shadow_test_index+1]} {luminance_cell_g_shadow_accumulated_path}' )
            await run( f'{python_command} {features_bases_dir}/plus_3d_data.py {tmp_luminance_cell_b_shadow_test_path} {luminance_cell_b_shadow_test_path[shadow_test_index+1]} {luminance_cell_b_shadow_accumulated_path}' )
        await run( f'{python_command} {features_bases_dir}/multiply_3d_data.py {luminance_cell_r_shadow_accumulated_path} {fpd_path} {luminance_r_fpd_path}' )
        await run( f'{python_command} {features_bases_dir}/multiply_3d_data.py {luminance_cell_g_shadow_accumulated_path} {fpd_path} {luminance_g_fpd_path}' )
        await run( f'{python_command} {features_bases_dir}/multiply_3d_data.py {luminance_cell_b_shadow_accumulated_path} {fpd_path} {luminance_b_fpd_path}' )
        await run( f'{python_command} {features_bases_dir}/line_of_sight_integration.py {compute_mode} {luminance_r_fpd_path} {depth_value_path} camera {xml_path1} {intensity_r_media_path}' )
        await run( f'{python_command} {features_bases_dir}/line_of_sight_integration.py {compute_mode} {luminance_g_fpd_path} {depth_value_path} camera {xml_path1} {intensity_g_media_path}' )
        await run( f'{python_command} {features_bases_dir}/line_of_sight_integration.py {compute_mode} {luminance_b_fpd_path} {depth_value_path} camera {xml_path1} {intensity_b_media_path}' )
        # surface luminance
        for shadow_test_index in range(len(shadow_test_path)):
            await run( f'{python_command} {features_bases_dir}/integral_cell.py {compute_mode} {extinction_path} light{shadow_test_index} {xml_path1} {integrated_extinction_cell_from_light_path[shadow_test_index]}' )
            await run( f'{python_command} {features_bases_dir}/convert_to_transmittance_cell.py {integrated_extinction_cell_from_light_path[shadow_test_index]} {transmittance_cell_from_light_path[shadow_test_index]}' )
        for shadow_test_index in range(len(shadow_test_path)-1):
            if shadow_test_index == 0:
                tmp_transmittance_cell_from_light_path = transmittance_cell_from_light_path[shadow_test_index]
            else:
                tmp_transmittance_cell_from_light_path = transmittance_cell_from_light_accumulated_path
            await run( f'{python_command} {features_bases_dir}/plus_3d_data.py {tmp_transmittance_cell_from_light_path} {transmittance_cell_from_light_path[shadow_test_index+1]} {transmittance_cell_from_light_accumulated_path}' )
        await run( f'{python_command} {features_bases_dir}/transmittance_surface_screen.py --extinction_path {extinction_path} --depth_value {depth_value_path} --xml {xml_path1} --output {transmittance_screen_media_path}' )
        await run( f'{python_command} {features_bases_dir}/multiply_2d_data.py {transmittance_screen_media_path} {intensity_r_surface_path} {intensity_r_transmittance_path}' )
        await run( f'{python_command} {features_bases_dir}/multiply_2d_data.py {transmittance_screen_media_path} {intensity_g_surface_path} {intensity_g_transmittance_path}' )
        await run( f'{python_command} {features_bases_dir}/multiply_2d_data.py {transmittance_screen_media_path} {intensity_b_surface_path} {intensity_b_transmittance_path}' )
        await run( f'{python_command} {features_bases_dir}/plus_2d_data.py {intensity_r_transmittance_path} {intensity_r_media_path} {intensity_r_path}' )
        await run( f'{python_command} {features_bases_dir}/plus_2d_data.py {intensity_g_transmittance_path} {intensity_g_media_path} {intensity_g_path}' )
        await run( f'{python_command} {features_bases_dir}/plus_2d_data.py {intensity_b_transmittance_path} {intensity_b_media_path} {intensity_b_path}' )

    await run( f'{python_command} {features_bases_dir}/convert_rgb_to_l.py {intensity_r_path} {intensity_g_path} {intensity_b_path} {intensity_l_path}' )
    # luminance basis
    await run( f'{python_command} {features_bases_dir}/gradient_feature.py {compute_mode} {intensity_l_path} {luminance_l_gradient_path}' )
    await run( f'{python_command} {features_bases_dir}/normalize_and_rotate_basis.py -i {luminance_l_gradient_path} -compute_mode {compute_mode} -out_norm {intensity_l_gradient_path} -out_perp {intensity_gradient_perp_path} -out_para {intensity_gradient_para_path}' )

    if False:
        # plot rgb image
        import matplotlib.pyplot as plt
        from util.features_bases.data_io import DataIO2D
        r = DataIO2D.InitLoadFile(intensity_r_path, taichi=False)
        g = DataIO2D.InitLoadFile(intensity_g_path, taichi=False)
        b = DataIO2D.InitLoadFile(intensity_b_path, taichi=False)
        rgb = np.stack([r.data_np, g.data_np, b.data_np], axis=-1)
        plt.imshow(rgb, origin='lower')
        plt.show()
        

    # apparent velocity
    if special_scene_type == special_scene_type_cloud:
        extinction_in_velocity = extinction_inside_path
        fpd_in_velocity = fpd_inside_path
    else:
        extinction_in_velocity = extinction_path
        fpd_in_velocity = fpd_path
    await run([ python_command, f'{features_bases_dir}/convert_world_velocity_to_screen_space.py', compute_mode, extinction_in_velocity, original_velocity_x_path, original_velocity_y_path, original_velocity_z_path, xml_path1, xml_path2, f'{fps}',  rel_velocity_screen_space_u_path, rel_velocity_screen_space_v_path ])
    await run([ python_command, f'{features_bases_dir}/multiply_3d_data.py', rel_velocity_screen_space_u_path, fpd_in_velocity, rel_velocity_screen_space_u_fpd_path ])
    await run([ python_command, f'{features_bases_dir}/multiply_3d_data.py', rel_velocity_screen_space_v_path, fpd_in_velocity, rel_velocity_screen_space_v_fpd_path ])
    if special_scene_type != special_scene_type_surface:
        await run([ python_command, f'{features_bases_dir}/line_of_sight_integration.py', compute_mode, rel_velocity_screen_space_u_fpd_path, '', 'camera', xml_path1, apparent_relative_velocity_u_path ])
        await run([ python_command, f'{features_bases_dir}/line_of_sight_integration.py', compute_mode, rel_velocity_screen_space_v_fpd_path, '', 'camera', xml_path1, apparent_relative_velocity_v_path ])
    else:
        # media apparent velocity
        await run([ python_command, f'{features_bases_dir}/line_of_sight_integration.py', compute_mode, rel_velocity_screen_space_u_fpd_path, depth_value_path, 'camera', xml_path1, apparent_relative_velocity_u_media_path ])
        await run([ python_command, f'{features_bases_dir}/line_of_sight_integration.py', compute_mode, rel_velocity_screen_space_v_fpd_path, depth_value_path, 'camera', xml_path1, apparent_relative_velocity_v_media_path ])
        # surface apparent velocity
        await run( f'{python_command} {features_bases_dir}/multiply_2d_data.py {apparent_relative_velocity_u_surface_path} {transmittance_screen_media_path} {apparent_relative_velocity_u_surface_transmittance_path}' )
        await run( f'{python_command} {features_bases_dir}/multiply_2d_data.py {apparent_relative_velocity_v_surface_path} {transmittance_screen_media_path} {apparent_relative_velocity_v_surface_transmittance_path}' )
        # combine
        await run( f'{python_command} {features_bases_dir}/plus_2d_data.py {apparent_relative_velocity_u_media_path} {apparent_relative_velocity_u_surface_transmittance_path} {apparent_relative_velocity_u_path}' )
        await run( f'{python_command} {features_bases_dir}/plus_2d_data.py {apparent_relative_velocity_v_media_path} {apparent_relative_velocity_v_surface_transmittance_path} {apparent_relative_velocity_v_path}' )

    # apparent velocity basis
    await run( f'{python_command} {features_bases_dir}/normalize_and_rotate_basis.py -i {apparent_relative_velocity_u_path} {apparent_relative_velocity_v_path} -compute_mode {compute_mode} -out_norm {apparent_relative_velocity_norm_path} -out_perp {apparent_relative_velocity_perp_path} -out_para {apparent_relative_velocity_para_path}' )

    # apparent normal
    if special_scene_type != special_scene_type_surface:
        await run( [ python_command, f'{features_bases_dir}/apparent_normal.py', compute_mode, transmittance_cell_path, fpd_path, '', xml_path1, world_scape_apparent_normal_path ])
        await run([ python_command, f'{features_bases_dir}/apparent_normal_world_to_view_transform.py', compute_mode, world_scape_apparent_normal_path, xml_path1, apparent_normal_x_path, apparent_normal_y_path, apparent_normal_z_path ])
    else:        
        await run( [ python_command, f'{features_bases_dir}/apparent_normal.py', compute_mode, transmittance_cell_path, fpd_path, depth_value_path, xml_path1, world_scape_apparent_normal_path ])
        await run([ python_command, f'{features_bases_dir}/apparent_normal_world_to_view_transform.py', compute_mode, world_scape_apparent_normal_path, xml_path1, apparent_normal_x_media_path, apparent_normal_y_media_path, apparent_normal_z_media_path ])
        # surface apparent normal
        await run( f'{python_command} {features_bases_dir}/multiply_2d_data.py {apparent_normal_x_surface_path} {transmittance_screen_media_path} {apparent_normal_x_surface_transmittance_path}' )
        await run( f'{python_command} {features_bases_dir}/multiply_2d_data.py {apparent_normal_y_surface_path} {transmittance_screen_media_path} {apparent_normal_y_surface_transmittance_path}' )
        await run( f'{python_command} {features_bases_dir}/multiply_2d_data.py {apparent_normal_z_surface_path} {transmittance_screen_media_path} {apparent_normal_z_surface_transmittance_path}' )
        # combine
        await run( f'{python_command} {features_bases_dir}/plus_2d_data.py {apparent_normal_x_media_path} {apparent_normal_x_surface_transmittance_path} {apparent_normal_x_path}' )
        await run( f'{python_command} {features_bases_dir}/plus_2d_data.py {apparent_normal_y_media_path} {apparent_normal_y_surface_transmittance_path} {apparent_normal_y_path}' )
        await run( f'{python_command} {features_bases_dir}/plus_2d_data.py {apparent_normal_z_media_path} {apparent_normal_z_surface_transmittance_path} {apparent_normal_z_path}' )

    # apparent normal basis
    await run( f'{python_command} {features_bases_dir}/normalize_and_rotate_basis.py -i {apparent_normal_x_path} {apparent_normal_y_path} -compute_mode {compute_mode} -out_perp {apparent_normal_perp_path} -out_para {apparent_normal_para_path}' )

    # curvature
    await run( f'{python_command} {features_bases_dir}/gaussian_filter_3d.py {transmittance_cell_path} {grid_gaussian_filter_sigma} {transmittance_cell_gaussian_filtered_path}' )
    if special_scene_type != special_scene_type_surface:
        await run([ python_command, f'{features_bases_dir}/curvature.py', compute_mode, transmittance_cell_gaussian_filtered_path, fpd_path, '', xml_path1, gaussian_curvature_path, mean_curvature_path ])
    else:
        await run([ python_command, f'{features_bases_dir}/curvature.py', compute_mode, transmittance_cell_gaussian_filtered_path, fpd_path, depth_value_path, xml_path1, gaussian_curvature_media_path, mean_curvature_media_path ])
        # surface curvature
        await run( f'{python_command} {features_bases_dir}/multiply_2d_data.py {gaussian_curvature_surface_path} {transmittance_screen_media_path} {gaussian_curvature_surface_transmittance_path}' )
        await run( f'{python_command} {features_bases_dir}/multiply_2d_data.py {mean_curvature_surface_path} {transmittance_screen_media_path} {mean_curvature_surface_transmittance_path}' )
        # combine
        await run( f'{python_command} {features_bases_dir}/plus_2d_data.py {gaussian_curvature_media_path} {gaussian_curvature_surface_transmittance_path} {gaussian_curvature_path}' )
        await run( f'{python_command} {features_bases_dir}/plus_2d_data.py {mean_curvature_media_path} {mean_curvature_surface_transmittance_path} {mean_curvature_path}' )


    # mean free path
    # await run( f'h5dump -d "resolution" {extinction_whole_path} && h5dump -d "resolution" {fpd_path} && stop' )
    if special_scene_type == special_scene_type_cloud:
        extinction_in_mean_free_path = extinction_whole_path
        fpd_in_mean_free_path = fpd_path
    else:
        extinction_in_mean_free_path = extinction_path
        fpd_in_mean_free_path = fpd_path
    await run( f'{python_command} {features_bases_dir}/invert_extinction.py --extinction {extinction_in_mean_free_path} --compute_mode {compute_mode} -o {mean_free_path_cell_path}' )
    await run( f'{python_command} {features_bases_dir}/multiply_3d_data.py {mean_free_path_cell_path} {fpd_in_mean_free_path} {mean_free_path_cell_fpd_path}' )
    if special_scene_type != special_scene_type_surface:
        await run( f'{python_command} {features_bases_dir}/line_of_sight_integration.py {compute_mode} {mean_free_path_cell_fpd_path} "" camera {xml_path1} {mean_free_path_path}' )
    else:
        await run( f'{python_command} {features_bases_dir}/line_of_sight_integration.py {compute_mode} {mean_free_path_cell_fpd_path} {depth_value_path} camera {xml_path1} {mean_free_path_path}' )
    # mean free path basis
    await run( f'{python_command} {features_bases_dir}/gradient_feature.py {compute_mode} {mean_free_path_path} {mean_free_path_gradient_path}' )
    await run( f'{python_command} {features_bases_dir}/normalize_and_rotate_basis.py -i {mean_free_path_gradient_path} -compute_mode {compute_mode} -out_perp {mean_free_path_gradient_perp_path} -out_para {mean_free_path_gradient_para_path}' )

    # temperature
    if original_temperature_path is not None:
        # surface temperature
        await run( f'{python_command} {features_bases_dir}/surface_and_background_temperature.py -depth {transmittance_screen_path} -back {background_temperature} -surf {surface_temperature} -out {temperature_surface_and_background_path}')
        await run( f'{python_command} {features_bases_dir}/multiply_2d_data.py {temperature_surface_and_background_path} {transmittance_screen_path} {temperature_surface_and_background_transmittance_path}' )
        # media temperature
        # temperature_depth_value_path = '""' if special_scene_type == special_scene_type_surface else 
        await run( f'{python_command} {features_bases_dir}/multiply_3d_data.py {original_temperature_path} {fpd_path} {temperature_fpd_path}' )
        await run( f'{python_command} {features_bases_dir}/line_of_sight_integration.py {compute_mode} {temperature_fpd_path} "" camera {xml_path1} {temperature_raw_media_path}' )
        await run( f'{python_command} {features_bases_dir}/temperature_scaler.py -i {temperature_raw_media_path} -min {media_temperature_min} -max {media_temperature_max} -o {temperature_media_path}' )
        await run( f'{python_command} {features_bases_dir}/plus_2d_data.py {temperature_surface_and_background_transmittance_path} {temperature_media_path} {temperature_path}' )

def xmlSettings(baseXMLPath, outputXMLPath, value):
    # print('baseXMLPath', baseXMLPath)
    tree = ET.parse(baseXMLPath)
    root = tree.getroot()
    for uni in value:
        if uni[0] == 'lights':
            light_index = uni[1]
            attribute = uni[2]
            value = uni[3]
            # print('light_index:', light_index)
            lights = root.find('lights').findall('light')
            # for index, light in enumerate(lights):
            #     print('light index', index)
            #     print('light tag', light.tag)
            #     print('light attrib', light.attrib)
            # print(f'num lights : {len(lights)}')
            if not 0 <= light_index < len(lights):
                raise IndexError(f'The light index ({light_index}) is not valid. The number of lights ')
            light = lights[light_index]
            # print( 'xml light', light.attrib )
            if attribute == 'color_multiply':
                # print('Updating color')
                color_multiply = float(value)
                # print('color_multiply:', color_multiply)
                # print( 'xml light color', light.get('color', None) )
                color = light.get('color', None).split(' ')
                if color is None:
                    raise ValueError(f'Light[{light_index}] does not have a "color" attribute.')
                # print('color', color)
                color = [float(c) * color_multiply for c in color]
                # print('color:', color)
                attribute = 'color'
                value = f"{color[0]} {color[1]} {color[2]}"
            # print('tag', light.tag)
            # print('attrib', light.attrib)
            # old_val = light.get(attribute, None)
            light.set(attribute, value)
            # print(f'Light[{light_index}] {attribute}: "{old_val}" → "{attribute}"')
            
        else:
            for name in root.iter(uni[0]):
                name.set(uni[1], uni[2]) 
    tree.write(outputXMLPath)

def blender_axis_to_opengl_axis(blender_axis):
    return [blender_axis[0], blender_axis[2], -blender_axis[1]]

def ComputeXML(scene_name, scene_json_settings, input_xml_path, frame_index, output_xml_path, print_one):
    # print('input_xml_path: ', input_xml_path, os.path.abspath(input_xml_path), os.path.exists(input_xml_path))
    # print('frame_index: ', frame_index)
    # print('output_xml_path: ', output_xml_path)
    
    changes_xml = []
    
    # resolution
    resolution = scene_json_settings['resolution']
    resolution = f"{resolution[0]} {resolution[1]}"
    changes_xml.append(  ["camera", "resolution", resolution]  )
    
    # camera
    camera_json_file_path = os.path.join(scene_name, 'assets', 'camera', f'camera_{frame_index:03d}.json')
    # print('camera_json_file_path', camera_json_file_path, os.path.abspath(camera_json_file_path), os.path.exists(camera_json_file_path) )
    if os.path.exists(camera_json_file_path):
        if print_one:
            print(f'Loading camera settings from {os.path.abspath(camera_json_file_path)}')
        camera_json = json.load(open(camera_json_file_path))
        camera_location = camera_json['camera']['location']
        camera_direction = camera_json['camera']['view_vector']
        camera_film_x = camera_json['camera']['film_x']
        camera_film_y = camera_json['camera']['film_y']
        camera_fov_x = camera_json['camera']['angle_x']
        camera_fov_y = camera_json['camera']['angle_y']
        # print('camera_location:', camera_location)
        changes_xml.extend([
            # ["camera", "position", list_to_vector_string(camera_location)],
            # ["camera", "direction", list_to_vector_string(camera_direction)],
            ["camera", "position", list_to_vector_string(blender_axis_to_opengl_axis(camera_location))],
            ["camera", "direction", list_to_vector_string(blender_axis_to_opengl_axis(camera_direction))],
            ["camera", "film", f"{camera_film_x} {camera_film_y}"],
            ["camera", "fov2", f"{camera_fov_x} {camera_fov_y}"]
        ])
    elif print_one:
        print(f'Camera animation is not available')
    
    # light
    light_json_file_path = os.path.join(scene_name, 'assets', 'light', f'light_{frame_index:03d}.json')
    # print('light_json_file_path', light_json_file_path, os.path.abspath(light_json_file_path), os.path.exists(light_json_file_path) )
    if os.path.exists( light_json_file_path ):
        if print_one:
            print(f'Loading light settings from {os.path.abspath(light_json_file_path)}')
        lights_json = json.load(open(light_json_file_path))['lights']
        for light_index, light in enumerate(lights_json):
            # print(light)
            # print(light['location'])
            # print(light['power'])
            changes_xml.extend([
                # [ "lights", light_index, "position", list_to_vector_string(light['location']) ],
                [ "lights", light_index, "position", list_to_vector_string(blender_axis_to_opengl_axis(light['location'])) ],
                [ "lights", light_index, "color_multiply", light['power'] ]
            ])
    elif print_one:
        print(f'Light animation is not available')
    
    xmlSettings(input_xml_path, output_xml_path, changes_xml)


def get_tmp_xml_path( scene_name, frame_index, suffix ):
    return os.path.join(scene_name, 'temp', 'tmp_xml', f'tmp_{frame_index:03d}_{suffix}.xml')

def prepare_xml( scene_name, scene_json_settings, frame_index, frame_end, print_one ):
    input_xml_path = os.path.join(scene_name, 'assets', 'scene_setting.xml')
    tmp1_file = get_tmp_xml_path(scene_name, frame_index, '1')
    tmp2_file = get_tmp_xml_path(scene_name, frame_index, '2')
    os.makedirs(os.path.dirname(tmp1_file), exist_ok=True)
    ComputeXML(scene_name, scene_json_settings, input_xml_path, frame_index, tmp1_file, print_one)
    ComputeXML(scene_name, scene_json_settings, input_xml_path, min(frame_index+1, frame_end), tmp2_file, False)

async def compute_features_for_target_frame( scene_name, settings, frame_index, sem, pbar ):
    try:
        async with sem:
            # print('compute frame_index:', frame_index)
        
            tmp1_file = get_tmp_xml_path( scene_name, frame_index, '1' )
            tmp2_file = get_tmp_xml_path( scene_name, frame_index, '2' )
            try:
                await compute_f_cs( scene_name, settings, frame_index, tmp1_file, tmp2_file )
            except Exception as e:
                return e
            # print('DEBUG : it will not remove temporary files')
            remove_tmp_frame( scene_name, settings, frame_index )

    finally:
        pbar.update(1)


async def compute_features_async( scene_name, scene_settings, range_frames, num_processes ):
    sem = asyncio.Semaphore(num_processes)
    pbar = tqdm(total=len(range_frames), desc="Features Processed", smoothing=0)

    tasks = []
    for frame_index in range_frames:
        task = asyncio.create_task(
            compute_features_for_target_frame(
                scene_name, scene_settings, frame_index, sem, pbar
            )
        )
        tasks.append(task)
    exceptions = await asyncio.gather(*tasks, return_exceptions=True)

    pbar.close()
    
    errors = []
    for idx, exc in enumerate(exceptions):
        if isinstance(exc, Exception):
            errors.append((idx, exc))
    if len(errors) > 0:
        for idx, err in errors:
            print('-'*10 + f"Error processing frame_index:{idx}" + '-'*10)
            if isinstance(err, ProcessExecutionError):
                return_code, cmd, stdout, stderr = err.args
                print('#'*4 , f"Return code: {return_code}", '#'*4)
                print('#'*4, f"Command:", '#'*4, f"\n{cmd}")
                print('#'*4, "STDOUT:", '#'*4, f"\n{stdout}")
                print('#'*4, "STDERR:", '#'*4, f"\n{stderr}")
                print('-'*20)
                # traceback.print_exception(err.__class__, err, err.__traceback__)
                print(''.join(traceback.format_exception(err.__class__, err, err.__traceback__)))
            elif isinstance(err, Exception):
                print(f"Exception: {err}")
                traceback.print_exception(err.__class__, err, err.__traceback__)
            else:
                print(f"Unknown error: {err}")
                print(f"Unknown error type: {type(err)}")
                traceback.print_exception(type(err), err, None)
    else:
        print("All frames processed successfully.")



def p2_compute_features( scene_name, json_fn, frame_start, frame_end, frame_skip):
    print(os.path.abspath(json_fn))
    with open( json_fn ) as f:
        scene_json_settings = json.load(f)
    prepare_internal_dirs( scene_name, scene_json_settings )
    
    if frame_start == frame_end:
        raise ValueError(f"Start frame ({frame_start}) cannot be equal to end frame ({frame_end}).")
    
    target_frames = unique_list( frame_start, frame_end, frame_skip )
    
    for i in target_frames:
        print_one = target_frames[0] == i
        prepare_xml( scene_name, scene_json_settings, i, frame_end , print_one=print_one )
    
    num_processes = 8
    asyncio.run( compute_features_async( scene_name, scene_json_settings, target_frames,  num_processes ))

    # input('DEBUG, temporary files are not removed, press Enter to continue...')
    print('DEBUG: it will not remove internal dirs')
    # remove_internal_dirs( scene_name, scene_settings )
        
    # run(f'{python_command} plot_f_cs.py {start_frame} -e {end_frame} -s {skip_frame} result features', shell=True)
