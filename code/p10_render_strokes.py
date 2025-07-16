# -----------------------------------------------------------------------------
# Stroke Transfer for Participating Media
# http://cg.it.aoyama.ac.jp/yonghao/sig25/abstsig25.html
#
# File: p10_render_strokes.py
# Maintainer: Hideki Todo
#
# Description:
# Render final stroke images with bump-mapped canvas and stroke textures as a simple lighting simulation.
# Wraps cpp_tools/bin/render_strokes_cli to produce high-quality stylized outputs.
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
import math
import os.path

from util.image.image2video import images_to_video
from util.pipeline.time_logger import gen_log_file_path
from util.pipeline.time_logger import log_timing, resume_log
from util.stroke_rendering.cpp_stroke_pipelines import render_strokes


def p10_render_strokes(
        frame_start=140,
        frame_end=151,
        frame_skip=1,
        num_textures=1,
        texture_length_mipmap_level=1,
        color_texture_filename="../../textures/texture.png",
        height_texture_filename="../../textures/height.png",
        out_final_filename_template="final/final_%03d.png",
        stroke_data_filename_template="stroke_data/stroke_%03d.h5",
        undercoat_filename_template="color/color_%03d.png",
        mask_file_template="",
        combine_height_top_factor=math.pow(2.0, 0.0),
        combine_height_additive_factor=math.pow(2.0, -2.0),
        combine_height_additive_log_factor=math.pow(2.0, 0.0),
        tex_step=math.pow(2.0, -7.953),
        height_scale=math.pow(2.0, -9.212),
        vz=3.2,
        lx=0.0,
        ly=1.2,
        lz=3.2,
        glossiness=math.pow(2.0, 8.130),
        kd=0.24,
        ks=0.01,
        ka=0.29,
        light_intensity=math.pow(2.0, 1.53),
        canvas_scale=0.4,
        resolution=(1024, 1024),
        out_video_final_file = None,
        frame_rate=24.0,
        log_file=None,
):
    """Render final stylized stroke images using bump-mapped canvas and lighting simulation.

    This function wraps the external binary `render_strokes_cli`
    to synthesize high-quality rendered images from stroke data.

    Args:
        frame_start (int): Start frame index.
        frame_end (int): End frame index.
        frame_skip (int): Frame sampling interval.
        num_textures (int): Number of textures used for stroke appearance.
        texture_length_mipmap_level (int): Mipmap level for stroke length and texture resolution.
        color_texture_filename (str): File path to the stroke color texture.
        height_texture_filename (str): File path to the canvas height (bump map) texture.
        out_final_filename_template (str): Output file template for rendered images.
        stroke_data_filename_template (str): Template for stroke data input (.h5).
        undercoat_filename_template (str): Template for undercoat (background color) image (.png).
        mask_file_template (str): Optional mask template for restricting stroke rendering.
        combine_height_top_factor (float): Weight for combining top canvas height.
        combine_height_additive_factor (float): Additive blending factor for height.
        combine_height_additive_log_factor (float): Log-space blending factor for height.
        tex_step (float): Step size for texture lookup.
        height_scale (float): Global scaling factor for bump map height.
        vz (float): Viewer's z-position for shading computation.
        lx (float): Light source x-position.
        ly (float): Light source y-position.
        lz (float): Light source z-position.
        glossiness (float): Glossiness coefficient (specular exponent).
        kd (float): Diffuse reflection coefficient.
        ks (float): Specular reflection coefficient.
        ka (float): Ambient reflection coefficient.
        light_intensity (float): Intensity of the simulated light source.
        canvas_scale (float): Scale factor for canvas shading effects.
        resolution (tuple): Output image resolution (width, height).
        out_video_final_file (str, optional): Output video file path. If provided, rendered frames are combined into a video.
        frame_rate (float): Frame rate for output video.
        log_file (str, optional): Path to the execution log file.
    """
    if log_file is None:
        out_dir = os.path.dirname(os.path.dirname(out_final_filename_template))
        log_name = os.path.join(out_dir, "log", "p10_render_strokes")
        log_file = gen_log_file_path(log_name)

    resume_log()
    render_strokes(frame_start=frame_start,
                   frame_end=frame_end,
                   frame_skip=frame_skip,
                   num_textures=num_textures,
                   texture_length_mipmap_level=texture_length_mipmap_level,
                   color_texture_filename=color_texture_filename,
                   height_texture_filename=height_texture_filename,
                   out_final_filename_template=out_final_filename_template,
                   stroke_data_filename_template=stroke_data_filename_template,
                   undercoat_filename_template=undercoat_filename_template,
                   mask_file_template=mask_file_template,
                   combine_height_top_factor=combine_height_top_factor,
                   combine_height_additive_factor=combine_height_additive_factor,
                   combine_height_additive_log_factor=combine_height_additive_log_factor,
                   tex_step=tex_step,
                   height_scale=height_scale,
                   vz=vz,
                   lx=lx,
                   ly=ly,
                   lz=lz,
                   glossiness=glossiness,
                   kd=kd,
                   ks=ks,
                   ka=ka,
                   light_intensity=light_intensity,
                   canvas_scale=canvas_scale,
                   resolution=resolution,
                   log_file=log_file
                   )
    log_timing("render_strokes", num_frames=len(range(frame_start, frame_end+1, frame_skip)))

    if out_video_final_file is not None:
        frames = range(frame_start, frame_end+1, frame_skip)
        images_to_video(out_final_filename_template, out_video_final_file, frame_rate=frame_rate, frames=frames)

    # delete stroke_data.h5
    # shutil.rmtree(os.path.dirname(stroke_data_filename_template), ignore_errors=True)
