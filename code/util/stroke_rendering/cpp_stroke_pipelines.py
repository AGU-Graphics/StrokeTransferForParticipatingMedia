# -----------------------------------------------------------------------------
# Stroke Transfer for Participating Media
# http://cg.it.aoyama.ac.jp/yonghao/sig25/abstsig25.html
#
# File: util/stroke_rendering/cpp_stroke_pipelines.py
# Maintainer: Hideki Todo and Yonghao Yue
#
# Description:
# Command-line wrapper for stroke generation and rendering using external C++ tools.
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
import os
import subprocess
import sys

_module_dir = os.path.dirname(os.path.abspath(__file__))
_code_dir = os.path.dirname(os.path.dirname(_module_dir))
_cpp_bin_dir = os.path.join(_code_dir, "cpp_tools", "bin")


def make_out_dir(file_path, base_dir="."):
    """Creates the output directory for the given file path if it doesn't exist.

    Args:
        file_path (str): Relative path to a target file.
        base_dir (str): Base directory to resolve relative paths.
    """
    out_dir = os.path.dirname(os.path.join(base_dir, file_path))
    os.makedirs(out_dir, exist_ok=True)


def tool_exe(tool_name):
    """Returns the full path to the specified CLI tool executable.

    Args:
        tool_name (str): Name of the executable.

    Returns:
        str: Full path to the CLI tool.
    """
    exe_file = os.path.join(_cpp_bin_dir, tool_name)
    return exe_file


def run_cli_command(tool_name, cmd):
    """Runs a CLI command and exits the program on failure.

    Args:
        tool_name (str): Name of the CLI tool (for logging).
        cmd (str): Command string to execute.
    """
    print(cmd)
    result = subprocess.run(cmd.split())
    if result.returncode != 0:
        print(f"Error running {tool_name}")
        sys.exit(1)


def gen_strokes(frame_start=140,
                frame_end=141,
                frame_skip=1,
                angular_random_offset_deg=5.0,
                random_offset_factor=1.0,
                length_factor=1.0,
                width_factor=1.0,
                length_random_factor_relative=0.0,
                width_random_factor_relative=0.0,
                sort_type="add_sort_index",
                clip_with_undercoat_alpha=0,
                texture_filename="assets/textures/texture.png",
                texture_for_active_set_for_new_stroke_filename="assets/textures/texture_for_new.png",
                texture_for_active_set_for_existing_stroke_filename="assets/textures/texture_for_existing.png",
                num_textures=1,
                texture_length_mipmap_level=1,
                out_anchor_filename_template="final/stroke_data/anchor/%3d.h5",
                out_stroke_filename_template="final/stroke_images/stroke/%03d.png",
                out_stroke_data_filename_template="final/stroke_data/stroke/%03d.h5",
                orientation_filename_template="temp/attributes/smooth_orientation/%03d.h5",
                undercoat_filename_template="temp/attributes/color/%03d.png",
                color_filename_template="temp/attributes/color/%03d.h5",
                width_filename_template="temp/attributes/width/%03d.h5",
                length_filename_template="temp/attributes/length/%03d.h5",
                frame_rate=24.0,
                resolution=(1024, 1024),
                mask_file_template="",
                active_set_file_template="",
                remove_hidden_strokes=1,
                remove_hidden_strokes_thr_contribution=1.0 / 256.0,
                remove_hidden_strokes_thr_alpha=1.0 / 256.0,
                stroke_dir="./",
                vx_filename_template="temp/raw_features/apparent_relative_velocity_u/%03d.h5",
                vy_filename_template="temp/raw_features/apparent_relative_velocity_v/%03d.h5",
                stroke_step_length=1.0 / 256.0,
                stroke_step_length_accuracy=0.1,
                consecutive_failure_max=100,
                log_file=None,
                region_label_file_template=None
                ):
    """Generate stroke anchor data, stroke images, and stroke metadata using a CLI tool.

    This function wraps an external C++ binary (gen_strokes_cli) to synthesize strokes
    based on orientation, color, width, and length attributes.

    Args:
        frame_start (int): Start frame index.
        frame_end (int): End frame index.
        frame_skip (int): Frame sampling interval.
        angular_random_offset_deg (float): Random angle perturbation in degrees.
        random_offset_factor (float): Scaling factor for random positional noise.
        length_factor (float): Global multiplier for stroke length.
        width_factor (float): Global multiplier for stroke width.
        length_random_factor_relative (float): Relative variation in stroke length.
        width_random_factor_relative (float): Relative variation in stroke width.
        sort_type (str): Sorting method for strokes (e.g., 'add_sort_index').
        clip_with_undercoat_alpha (int): Whether to use undercoat alpha for clipping (0 or 1).
        texture_filename (str): Path to primary stroke texture image.
        texture_for_active_set_for_new_stroke_filename (str): Texture for new strokes.
        texture_for_active_set_for_existing_stroke_filename (str): Texture for existing strokes.
        num_textures (int): Number of textures to apply.
        texture_length_mipmap_level (int): Mipmap level used for stroke texturing.
        out_anchor_filename_template (str): Output filename template for stroke anchor files (.h5).
        out_stroke_filename_template (str): Output filename template for stroke images (.png).
        out_stroke_data_filename_template (str): Output filename template for stroke data (.h5).
        orientation_filename_template (str): Template for input orientation field (.h5).
        undercoat_filename_template (str): Template for undercoat color image (.png).
        color_filename_template (str): Template for stroke color attributes (.h5).
        width_filename_template (str): Template for stroke width attributes (.h5).
        length_filename_template (str): Template for stroke length attributes (.h5).
        frame_rate (float): Frame rate for video (used for animation alignment).
        resolution (tuple): Output image resolution as (width, height).
        mask_file_template (str): Optional template for alpha or spatial mask.
        active_set_file_template (str): Optional template for active stroke set mask.
        remove_hidden_strokes (int): Whether to discard strokes occluded by undercoat (0 or 1).
        remove_hidden_strokes_thr_contribution (float): Threshold on undercoat contribution.
        remove_hidden_strokes_thr_alpha (float): Threshold on undercoat alpha.
        stroke_dir (str): Base directory for stroke outputs.
        vx_filename_template (str): Template for velocity field (u-component).
        vy_filename_template (str): Template for velocity field (v-component).
        stroke_step_length (float): Integration step size for stroke tracing.
        stroke_step_length_accuracy (float): Accuracy for step size estimation.
        consecutive_failure_max (int): Max number of consecutive tracing failures allowed.
        log_file (str, optional): Path to output log file.
        region_label_file_template (str, optional): Template for region label files.
    """
    width, height = resolution

    exe_file = tool_exe("gen_strokes_cli")

    make_out_dir(out_stroke_filename_template)
    make_out_dir(out_anchor_filename_template)
    make_out_dir(out_stroke_data_filename_template)

    if log_file is not None:
        os.makedirs(os.path.dirname(log_file), exist_ok=True)

    cmd = exe_file
    cmd += f" --width {width}"
    cmd += f" --height {height}"
    cmd += f" --frame_start {frame_start}"
    cmd += f" --frame_end {frame_end}"
    cmd += f" --frame_skip {frame_skip}"
    cmd += f" --angular_random_offset_deg {angular_random_offset_deg}"
    cmd += f" --random_offset_factor {random_offset_factor}"
    cmd += f" --length_factor {length_factor}"
    cmd += f" --width_factor {width_factor}"
    cmd += f" --length_random_factor_relative {length_random_factor_relative}"
    cmd += f" --width_random_factor_relative {width_random_factor_relative}"
    cmd += f" --stroke_step_length {stroke_step_length}"
    cmd += f" --stroke_step_length_accuracy {stroke_step_length_accuracy}"
    cmd += f" --consecutive_failure_max {consecutive_failure_max}"
    cmd += f" --sort_type {sort_type}"
    cmd += f" --clip_with_undercoat_alpha {clip_with_undercoat_alpha}"
    cmd += f" --texture_filename {texture_filename}"
    cmd += f" --texture_for_active_set_for_new_stroke_filename {texture_for_active_set_for_new_stroke_filename}"
    cmd += f" --texture_for_active_set_for_existing_stroke_filename {texture_for_active_set_for_existing_stroke_filename}"
    cmd += f" --num_textures {num_textures}"
    cmd += f" --texture_length_mipmap_level {texture_length_mipmap_level}"
    cmd += f" --out_anchor_filename_template {out_anchor_filename_template}"
    cmd += f" --out_stroke_filename_template {out_stroke_filename_template}"
    cmd += f" --out_stroke_data_filename_template {out_stroke_data_filename_template}"
    cmd += f" --orientation_filename_template {orientation_filename_template}"
    cmd += f" --undercoat_filename_template {undercoat_filename_template}"
    cmd += f" --color_filename_template {color_filename_template}"
    cmd += f" --width_filename_template {width_filename_template}"
    cmd += f" --length_filename_template {length_filename_template}"
    cmd += f" --vx_filename_template {vx_filename_template}"
    cmd += f" --vy_filename_template {vy_filename_template}"
    if region_label_file_template is not None:
        cmd += f" --region_label_template {region_label_file_template}"
    cmd += f" --frame_rate {frame_rate}"
    if mask_file_template != "":
        cmd += f" --mask_file_template {mask_file_template}"
    if active_set_file_template != "":
        cmd += f" --active_set_file_template {active_set_file_template}"
    cmd += f" --remove_hidden_strokes {remove_hidden_strokes}"
    cmd += f" --remove_hidden_strokes_thr_contribution {remove_hidden_strokes_thr_contribution}"
    cmd += f" --remove_hidden_strokes_thr_alpha {remove_hidden_strokes_thr_alpha}"

    if log_file is not None:
        cmd += f" --log_file_name {log_file}"
    else:
        cmd += f" --log_file_name gen_strokes.log"

    run_cli_command("genstroke_cli", cmd)


def render_strokes(frame_start=140,
                   frame_end=151,
                   frame_skip=1,
                   num_textures=1,
                   texture_length_mipmap_level=1,
                   color_texture_filename="assets/textures/texture.png",
                   height_texture_filename="assets/textures/height.png",
                   out_final_filename_template="final/stroke_image/final/%03d.png",
                   stroke_data_filename_template="final/stroke_data/stroke/%03d.h5",
                   undercoat_filename_template="temp/attributes/color/%03d.png",
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
                   stroke_dir="./",
                   log_file=None
                   ):
    """Renders strokes into final images using lighting and height texture via CLI.

    Args:
        frame_start (int): Start frame index.
        frame_end (int): End frame index.
        color_texture_filename (str): Path to color texture image.
        height_texture_filename (str): Path to height texture image.
        stroke_data_filename_template (str): Input stroke data file template.
        out_final_filename_template (str): Output template for rendered images.
        resolution (tuple): Output resolution.
        log_file (str, optional): Path to log output file.
    """
    width, height = resolution

    exe_file = tool_exe("render_strokes_cli")

    make_out_dir(out_final_filename_template)

    if log_file is not None:
        os.makedirs(os.path.dirname(log_file), exist_ok=True)

    cmd = exe_file
    cmd += f" --width {width}"
    cmd += f" --height {height}"
    cmd += f" --stroke_dir {stroke_dir}"
    cmd += f" --frame_start {frame_start}"
    cmd += f" --frame_end {frame_end}"
    cmd += f" --frame_skip {frame_skip}"
    cmd += f" --color_texture_filename {color_texture_filename}"
    cmd += f" --height_texture_filename {height_texture_filename}"
    cmd += f" --num_textures {num_textures}"
    cmd += f" --texture_length_mipmap_level {texture_length_mipmap_level}"
    cmd += f" --stroke_data_filename_template {stroke_data_filename_template}"
    cmd += f" --out_final_filename_template {out_final_filename_template}"
    cmd += f" --undercoat_filename_template {undercoat_filename_template}"
    cmd += f" --combine_height_top_factor {combine_height_top_factor}"
    cmd += f" --combine_height_additive_factor {combine_height_additive_factor}"
    cmd += f" --combine_height_additive_log_factor {combine_height_additive_log_factor}"
    cmd += f" --tex_step {tex_step}"
    cmd += f" --height_scale {height_scale}"
    cmd += f" --vz {vz}"
    cmd += f" --lx {lx}"
    cmd += f" --ly {ly}"
    cmd += f" --lz {lz}"
    cmd += f" --glossiness {glossiness}"
    cmd += f" --kd {kd}"
    cmd += f" --ks {ks}"
    cmd += f" --ka {ka}"
    cmd += f" --light_intensity {light_intensity}"
    cmd += f" --canvas_scale {canvas_scale}"

    if log_file is not None:
        cmd += f" --log_file_name {log_file}"
    else:
        cmd += f" --log_file_name render_strokes.log"
    if mask_file_template != "":
        cmd += f" --mask_file_template {mask_file_template}"

    run_cli_command("render_strokes_cli", cmd)


def render_strokes_pencil(
        frame_start=140,
        frame_end=151,
        frame_skip=1,
        num_textures=1,
        texture_length_mipmap_level=1,
        color_texture_filename="assets/textures/pencil_texture.png",
        out_final_filename_template="final/stroke_image/final/%03d.png",
        stroke_data_filename_template="final/stroke_data/stroke/%03d.h5",
        undercoat_filename_template="temp/attributes/color/%03d.png",
        mask_file_template="",
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
        stroke_dir="./",
        pencil_factor=0.5,
        paper_texture_filename="assets/textures/paper_texture.png",
        tex_step=1.0,
        log_file=None
):
    """Renders pencil-style strokes using paper texture and shading parameters.

    Args:
        frame_start (int): Start frame index.
        frame_end (int): End frame index.
        color_texture_filename (str): Path to pencil-style texture.
        paper_texture_filename (str): Background paper texture file.
        pencil_factor (float): Blending weight for pencil shading.
        stroke_data_filename_template (str): Input stroke data file template.
        out_final_filename_template (str): Output template for final images.
        resolution (tuple): Output resolution.
        log_file (str, optional): Path to log output file.
    """
    width, height = resolution

    exe_file = tool_exe("render_strokes_pencil_cli")

    make_out_dir(out_final_filename_template)

    if log_file is not None:
        os.makedirs(os.path.dirname(log_file), exist_ok=True)

    tex_step_x = tex_step / width
    tex_step_y = tex_step / height

    cmd = exe_file

    cmd += f" --width {width}"
    cmd += f" --height {height}"
    cmd += f" --stroke_dir {stroke_dir}"
    cmd += f" --frame_start {frame_start}"
    cmd += f" --frame_end {frame_end}"
    cmd += f" --frame_skip {frame_skip}"
    cmd += f" --color_texture_filename {color_texture_filename}"
    cmd += f" --paper_texture_filename {paper_texture_filename}"
    cmd += f" --num_textures {num_textures}"
    cmd += f" --texture_length_mipmap_level {texture_length_mipmap_level}"
    cmd += f" --stroke_data_filename_template {stroke_data_filename_template}"
    cmd += f" --out_final_filename_template {out_final_filename_template}"
    cmd += f" --undercoat_filename_template {undercoat_filename_template}"
    cmd += f" --tex_step_x {tex_step_x}"
    cmd += f" --tex_step_y {tex_step_y}"
    cmd += f" --height_scale {height_scale}"
    cmd += f" --vz {vz}"
    cmd += f" --lx {lx}"
    cmd += f" --ly {ly}"
    cmd += f" --lz {lz}"
    cmd += f" --glossiness {glossiness}"
    cmd += f" --kd {kd}"
    cmd += f" --ks {ks}"
    cmd += f" --ka {ka}"
    cmd += f" --light_intensity {light_intensity}"
    cmd += f" --pencil_factor {pencil_factor}"
    cmd += f" --canvas_scale {canvas_scale}"
    if mask_file_template != "":
        cmd += f" --mask_file_template {mask_file_template}"

    if log_file is not None:
        cmd += f" --log_file_name {log_file}"
    else:
        cmd += f" --log_file_name render_strokes_pencil.log"

    run_cli_command("render_strokes_pencil_cli", cmd)


if __name__ == '__main__':
    print(f"{_code_dir}: {os.path.exists(_code_dir)}")
    print(f"{_cpp_bin_dir}: {os.path.exists(_cpp_bin_dir)}")
