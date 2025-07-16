# -----------------------------------------------------------------------------
# Stroke Transfer for Participating Media
# http://cg.it.aoyama.ac.jp/yonghao/sig25/abstsig25.html
#
# File: p10_render_strokes_pencil.py
# Maintainer: Hideki Todo
#
# Description:
# Render pencil-style stroke images with lighting and paper texture effects.
# Wraps cpp_tools/bin/render_strokes_pencil_cli to simulate hand-drawn appearance.
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
from util.stroke_rendering.cpp_stroke_pipelines import render_strokes_pencil


def p10_render_strokes_pencil(
        frame_start=140,
        frame_end=151,
        frame_skip=1,
        num_textures=1,
        texture_length_mipmap_level=1,
        color_texture_filename="../../textures/texture.png",
        out_final_filename_template="final/final_%03d.png",
        stroke_data_filename_template="stroke_data/stroke_%03d.h5",
        undercoat_filename_template="color/color_%03d.png",
        mask_file_template="",
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
        out_video_final_file=None,
        frame_rate=24.0,
        log_file=None,
        pencil_factor=0.5,
        paper_texture_filename="../../textures/paper1024.png",
):
    """Render pencil-style stroke images using lighting and paper texture effects.

    This function wraps the external CLI tool `render_strokes_pencil_cli`
    to generate hand-drawn-style stroke renderings.

    Args:
        frame_start (int): Start frame index.
        frame_end (int): End frame index.
        frame_skip (int): Interval between frames to process.
        num_textures (int): Number of stroke textures to cycle through.
        texture_length_mipmap_level (int): Mipmap level for stroke texture resolution.
        color_texture_filename (str): File path to the stroke texture image (grayscale or color).
        out_final_filename_template (str): Output path template for rendered final images.
        stroke_data_filename_template (str): Template for stroke data input (.h5 files).
        undercoat_filename_template (str): Template for undercoat background images.
        mask_file_template (str): Optional mask to restrict stroke rendering.
        tex_step (float): Texture sampling step size.
        height_scale (float): Scale factor for canvas bump height.
        vz (float): Z-position of the viewer camera for shading.
        lx (float): X-position of the light source.
        ly (float): Y-position of the light source.
        lz (float): Z-position of the light source.
        glossiness (float): Glossiness coefficient for specular highlights.
        kd (float): Diffuse reflection coefficient.
        ks (float): Specular reflection coefficient.
        ka (float): Ambient reflection coefficient.
        light_intensity (float): Intensity of the simulated directional light.
        canvas_scale (float): Scaling factor for shading on the canvas.
        resolution (tuple): Output image resolution as (width, height).
        out_video_final_file (str, optional): Output video path. If provided, frames are encoded into a video.
        frame_rate (float): Frame rate for the output video.
        log_file (str, optional): Path to the execution log file.
        pencil_factor (float): Intensity of the pencil-style effect (range 0–1).
        paper_texture_filename (str): File path to the background paper texture image.
    """
    if log_file is None:
        out_dir = os.path.dirname(os.path.dirname(out_final_filename_template))
        log_name = os.path.join(out_dir, "log", "p10_render_strokes_pencil")
        log_file = gen_log_file_path(log_name)

    render_strokes_pencil(
        frame_start=frame_start,
        frame_end=frame_end,
        frame_skip=frame_skip,
        num_textures=num_textures,
        texture_length_mipmap_level=texture_length_mipmap_level,
        color_texture_filename=color_texture_filename,
        out_final_filename_template=out_final_filename_template,
        stroke_data_filename_template=stroke_data_filename_template,
        undercoat_filename_template=undercoat_filename_template,
        mask_file_template=mask_file_template,
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
        log_file=log_file,
        pencil_factor=pencil_factor,
        paper_texture_filename=paper_texture_filename,
    )

    if out_video_final_file is not None:
        frames = range(frame_start, frame_end + 1, frame_skip)
        images_to_video(out_final_filename_template, out_video_final_file, frame_rate=frame_rate, frames=frames)

    # delete stroke_data.h5
    # shutil.rmtree(os.path.dirname(stroke_data_filename_template), ignore_errors=True)
