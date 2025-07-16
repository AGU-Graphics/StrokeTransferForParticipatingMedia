# -----------------------------------------------------------------------------
# Stroke Transfer for Participating Media
# http://cg.it.aoyama.ac.jp/yonghao/sig25/abstsig25.html
#
# File: util/relocation/relocation.py
# Maintainer: Yonghao Yue and Hideki Todo
#
# Description:
# Procedural relocation (image displacement) using fractal and white noise fields.
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
import os.path

import cv2
import h5py
import noise
import numpy as np
import tqdm

from util.common.feature_basis_io import load_rgba, save_hdf5, save_rgba, load_image, save_image
from util.infra.logger import getLogger

logger = getLogger()


def load_raw_hdf5(in_file):
    """
    Load raw HDF5 file with "shape" and "data" keys.

    Args:
        in_file (str): Path to input HDF5 file.

    Returns:
        np.ndarray: Loaded data.
    """
    with h5py.File(in_file, mode='r') as f:
        shape = np.array(f["shape"][()]).flatten()
        data = np.array(f["data"][()])
    return data


def save_raw_hdf5(out_file, data):
    """
    Save a NumPy array to HDF5 with "shape" and "data" keys.

    Args:
        out_file (str): Output file path.
        data (np.ndarray): Data to be saved.
    """
    with h5py.File(out_file, mode='w') as f:
        shape = np.array(data.shape)
        f.create_dataset("shape", data=shape, compression="gzip")
        f.create_dataset("data", data=data, compression="gzip")


def rescale_noise(noise):
    """
    Rescale noise values to [0, 1] based on symmetric min/max.

    Args:
        noise (np.ndarray): Input noise array.

    Returns:
        np.ndarray: Rescaled noise.
    """
    _min = np.min(noise)
    _max = np.max(noise)
    _s = np.max([np.abs(_min), np.abs(_max)])
    return (noise / _s + 1.0) * 0.5


def generate_fractal_noise(shape, shift_x, shift_y, scale, persistence, lacunarity, octaves):
    """
    Generate 2D fractal noise using Perlin noise.

    Args:
        shape (tuple): Output (height, width).
        shift_x (float): X-axis offset.
        shift_y (float): Y-axis offset.
        scale (float): Frequency scale.
        persistence (float): Persistence value.
        lacunarity (float): Lacunarity value.
        octaves (int): Number of noise octaves.

    Returns:
        np.ndarray: 2D fractal noise.
    """
    res_y, res_x = shape
    fractal_noise = np.zeros(shape)
    for i in range(shape[0]):
        for j in range(shape[1]):
            fractal_noise[i][j] = noise.pnoise3((i + shift_y) / scale, (j + shift_x) / scale, 0,
                                                octaves=octaves, persistence=persistence, lacunarity=lacunarity,
                                                repeatx=res_x, repeaty=res_y, base=0)
    return rescale_noise(fractal_noise)


def generate_3d_fractal_noise(shape, shift_x, shift_y, shift_z, scale_space, scale_time, persistence, lacunarity,
                              octaves):
    """
    Generate 3D fractal noise for volume displacement.

    Args:
        shape (tuple): (depth, height, width).
        shift_x, shift_y, shift_z (float): Spatial offsets.
        scale_space (float): Spatial frequency scale.
        scale_time (float): Temporal frequency scale.
        persistence (float): Persistence.
        lacunarity (float): Lacunarity.
        octaves (int): Number of octaves.

    Returns:
        np.ndarray: 3D noise volume.
    """
    res_z, res_y, res_x = shape
    fractal_noise = np.zeros(shape)

    res_z_ref = 180
    res_xy_ref = 512

    z_scale = shape[0] / res_z_ref
    xy_scale = max(res_x, res_y) / res_xy_ref

    print(shift_z / scale_time / z_scale)

    for i in range(shape[0]):
        for j in range(shape[1]):
            for k in range(shape[2]):
                fractal_noise[i][j][k] = noise.pnoise3((i + shift_z) / scale_time / z_scale,
                                                       (j + shift_y) / scale_space / xy_scale,
                                                       (k + shift_x) / scale_space / xy_scale,
                                                       octaves=octaves, persistence=persistence, lacunarity=lacunarity,
                                                       repeatx=res_x, repeaty=res_y, repeatz=res_z,
                                                       base=0)
    return rescale_noise(fractal_noise)


def generate_fractal_noise_frame(
        frame,
        resolution,
        shift_x, shift_y, shift_z,
        scale_space, scale_time, persistence, lacunarity, octaves,
        repeat_xy=128,
        repeat_z=128):
    """
    Generate single-frame 2D fractal noise for displacement.

    Args:
        frame (int): Frame index.
        resolution (tuple): (width, height).
        shift_x, shift_y, shift_z (float): Spatial/temporal offsets.
        scale_space, scale_time (float): Frequency scales.
        persistence, lacunarity (float): Noise parameters.
        octaves (int): Octave count.
        repeat_xy, repeat_z (int): Repetition period.

    Returns:
        np.ndarray: Noise image.
    """
    # print(f"- scale_space: {scale_space}")
    # print(f"- scale_time: {scale_time}")
    res_x, res_y = resolution
    res_xy_ref = 512
    res_z_ref = 180

    shape = res_y, res_x

    fractal_noise = np.zeros(shape)

    z_scale = 1
    xy_scale = max(res_x, res_y) / res_xy_ref

    for y in range(shape[0]):
        for x in range(shape[1]):
            fractal_noise[y][x] = noise.pnoise3((frame + shift_z) / scale_time / z_scale,
                                                (y + shift_y) / scale_space / xy_scale,
                                                (x + shift_x) / scale_space / xy_scale,
                                                octaves=octaves, persistence=persistence, lacunarity=lacunarity,
                                                repeatx=repeat_xy, repeaty=repeat_xy, repeatz=repeat_z,
                                                base=0)
    return rescale_noise(fractal_noise)


def generate_octave_noise_frames(
        out_noise_file="relocate/octave_noise.hdf5",
        out_displacement_file="relocate/octave_displacement.hdf5",
        frames=[1],
        resolution=(512, 512),
        r_scale=0.005,
        scale_space=25,
        scale_time=0.01,
        persistence=5 / 10,
        lacunarity=20 / 10,
        octaves=7,
        shift_x1=48,
        shift_y1=32,
        shift_z1=40,
        shift_x2=175,
        shift_y2=192,
        shift_z2=180):
    """
    Generate multi-frame octave noise and save displacement vectors.

    Args:
        out_noise_file (str): Path to save raw noise.
        out_displacement_file (str): Path to save displacement.
        frames (List[int]): Frame indices.
        resolution (tuple): (width, height).
        r_scale (float): Scale factor for displacement.
        scale_space, scale_time, persistence, lacunarity, octaves (float/int): Noise parameters.
        shift_x1, shift_y1, shift_z1 (float): Shift for first direction.
        shift_x2, shift_y2, shift_z2 (float): Shift for second direction.
    """
    res_x, res_y = resolution
    res_z = len(frames)

    max_res = max([res_x, res_y])

    shape = res_z, res_y, res_x

    fractal_noise_r = generate_3d_fractal_noise(shape, 0, 0, 0, scale_space, scale_time, persistence, lacunarity,
                                                octaves)
    print("fractal_noise_r: ", np.min(fractal_noise_r), "-", np.max(fractal_noise_r))

    fractal_noise_t1 = generate_3d_fractal_noise(shape, shift_x1, shift_y1, shift_z1, scale_space, scale_time,
                                                 persistence, lacunarity,
                                                 octaves)
    print("fractal_noise_t1: ", np.min(fractal_noise_t1), "-", np.max(fractal_noise_t1))

    fractal_noise_t2 = generate_3d_fractal_noise(shape, shift_x2, shift_y2, shift_z2, scale_space, scale_time,
                                                 persistence, lacunarity,
                                                 octaves)
    print("fractal_noise_t2: ", np.min(fractal_noise_t2), "-", np.max(fractal_noise_t2))

    noise_data = np.array([fractal_noise_r, fractal_noise_t1, fractal_noise_t2])

    out_dir = os.path.dirname(out_noise_file)
    os.makedirs(out_dir, exist_ok=True)
    save_raw_hdf5(out_noise_file, noise_data)

    _ux_octave = fractal_noise_t1 - 0.5
    _uy_octave = fractal_noise_t2 - 0.5
    norm = np.sqrt(_ux_octave * _ux_octave + _uy_octave * _uy_octave) + 1.0e-5
    ux_octave = r_scale * fractal_noise_r * _ux_octave / norm
    uy_octave = r_scale * fractal_noise_r * _uy_octave / norm

    displacement_data = np.array([ux_octave, uy_octave])
    out_dir = os.path.dirname(out_displacement_file)
    os.makedirs(out_dir, exist_ok=True)
    save_raw_hdf5(out_displacement_file, displacement_data)


def generate_white_noise_frames(
        out_noise_file="relocate/white_noise.hdf5",
        out_displacement_file="relocate/white_displacement.hdf5",
        frames=[1],
        resolution=(512, 512),
        r_scale=0.005):
    """
    Generate white noise and save radial displacement vectors.

    Args:
        out_noise_file (str): Path to save raw noise.
        out_displacement_file (str): Path to save displacement.
        frames (List[int]): Frame indices.
        resolution (tuple): (width, height).
        r_scale (float): Scale factor for displacement.
    """
    res_x, res_y = resolution
    res_z = len(frames)

    max_res = max([res_x, res_y])

    shape = res_z, res_y, res_x

    rng = np.random.default_rng()
    white_r = rng.random(shape)
    white_t = rng.random(shape)

    noise_data = np.array([white_r, white_t])

    out_dir = os.path.dirname(out_noise_file)
    os.makedirs(out_dir, exist_ok=True)
    save_raw_hdf5(out_noise_file, noise_data)

    ux_white = r_scale * white_r * np.cos(white_t * 2.0 * np.pi)
    uy_white = r_scale * white_r * np.sin(white_t * 2.0 * np.pi)

    displacement_data = np.array([ux_white, uy_white])
    out_dir = os.path.dirname(out_displacement_file)
    os.makedirs(out_dir, exist_ok=True)
    save_raw_hdf5(out_displacement_file, displacement_data)


def relocate_by_displacement_frames(
        input_file_template, output_file_templates,
        displacement_file,
        frames=[1],
        resolution=None):
    """
    Apply displacement maps to a sequence of frames.

    Args:
        input_file_template (str): Input file path template.
        output_file_templates (List[str]): Output path templates.
        displacement_file (str): HDF5 with [ux, uy] displacements.
        frames (List[int]): Frame indices.
        resolution (tuple): Optional image resolution.
    """
    if resolution is None:
        try:
            input_file = input_file_template % frames[0]
        except:
            input_file = input_file_template

        I = load_image(input_file)
        res_y, res_x = I.shape[:2]
    else:
        res_x, res_y = resolution
    res_z = len(frames)

    max_res = max([res_x, res_y])

    shape = res_z, res_y, res_x

    uxy = load_raw_hdf5(displacement_file)

    print(f"resolution: {resolution}")

    print(f"uxy: {uxy.shape}")

    ux, uy = uxy

    print(f"ux: {ux.shape}")
    print(f"uy: {uy.shape}")

    orig_x = np.arange(res_x)
    orig_y = np.arange(res_y)

    PX, PY = np.meshgrid(orig_x, orig_y)

    for i, frame in enumerate(tqdm.tqdm(frames)):
        try:
            input_file = input_file_template % frame
        except:
            input_file = input_file_template

        I = load_image(input_file)

        ux_i = cv2.resize(ux[i], (res_x, res_y))
        uy_i = cv2.resize(uy[i], (res_x, res_y))

        SPX_OCT = np.clip((PX + max_res * ux_i).astype(int), 0, res_x - 1)
        SPY_OCT = np.clip((PY + max_res * uy_i).astype(int), 0, res_y - 1)

        if I.ndim == 3:
            O_oct = I[SPY_OCT, SPX_OCT, :]
        elif I.ndim == 2:
            O_oct = I[SPY_OCT, SPX_OCT]

        for output_file_template in output_file_templates:
            output_file = output_file_template % frame
            out_dir = os.path.dirname(output_file)
            os.makedirs(out_dir, exist_ok=True)
            save_image(output_file, O_oct)


def relocate_by_octave_noise_file(
        input_file, output_file,
        resolution=None,
        r_scale=30 / 1000,
        scale=25,
        persistence=5 / 10,
        lacunarity=20 / 10,
        octaves=7,
        shift_x1=48,
        shift_y1=32,
        shift_x2=175,
        shift_y2=192,
):
    """
    Apply single-frame octave noise to an image.

    Args:
        input_file (str): Input image path.
        output_file (str): Output image path.
        resolution (tuple): Optional image resolution.
        r_scale, scale, persistence, lacunarity, octaves (float/int): Noise parameters.
        shift_x1, shift_y1, shift_x2, shift_y2 (float): Shifts for noise vectors.
    """
    I = load_image(input_file)

    if resolution is None:
        res_y, res_x = I.shape[:2]
        resolution = (res_x, res_y)
    else:
        res_x, res_y = resolution

    max_res = max([res_x, res_y])

    x1 = (np.arange(res_x) + 0.5) / max_res
    x2 = (np.arange(res_y) + 0.5) / max_res

    X1, X2 = np.meshgrid(x1, x2)

    shape = (res_y, res_x)

    fractal_noise_r = generate_fractal_noise(shape, 0, 0, scale, persistence, lacunarity, octaves)
    print("fractal_noise_r: ", np.min(fractal_noise_r), "-", np.max(fractal_noise_r))

    fractal_noise_t1 = generate_fractal_noise(shape, shift_x1, shift_y1, scale, persistence, lacunarity, octaves)
    print("fractal_noise_t1: ", np.min(fractal_noise_t1), "-", np.max(fractal_noise_t1))

    fractal_noise_t2 = generate_fractal_noise(shape, shift_x2, shift_y2, scale, persistence, lacunarity, octaves)
    print("fractal_noise_t2: ", np.min(fractal_noise_t2), "-", np.max(fractal_noise_t2))

    _ux_octave = fractal_noise_t1 - 0.5
    _uy_octave = fractal_noise_t2 - 0.5
    norm = np.sqrt(_ux_octave * _ux_octave + _uy_octave * _uy_octave) + 1.0e-5
    ux_octave = r_scale * fractal_noise_r * _ux_octave / norm
    uy_octave = r_scale * fractal_noise_r * _uy_octave / norm

    orig_x = np.arange(res_x)
    orig_y = np.arange(res_y)

    PX, PY = np.meshgrid(orig_x, orig_y)

    SPX_OCT = np.clip((PX + max_res * ux_octave).astype(int), 0, res_x - 1)
    SPY_OCT = np.clip((PY + max_res * uy_octave).astype(int), 0, res_y - 1)
    O_oct = I[SPY_OCT, SPX_OCT, :]

    save_image(output_file, O_oct)

    save_image(output_file.replace(".png", ".hdf5"), np.flipud(O_oct))


def relocate_by_octave_noise_frames(
        input_file_template, output_file_template,
        frames=[1],
        resolution=None,
        r_scale=0.01,
        scale=25,
        persistence=5 / 10,
        lacunarity=20 / 10,
        octaves=7,
        shift_x1=48,
        shift_y1=32,
        shift_z1=40,
        shift_x2=175,
        shift_y2=192,
        shift_z2=180):
    """
    Apply octave noise to a series of frames.

    Args:
        input_file_template (str): Template for input file paths.
        output_file_template (str): Template for output paths.
        frames (List[int]): Frame indices.
        resolution (tuple): Optional image resolution.
        r_scale, scale, persistence, lacunarity, octaves (float/int): Noise parameters.
        shift_* (float): Shift values for noise vectors.
    """
    if resolution is None:
        try:
            input_file = input_file_template % frames[0]
        except:
            input_file = input_file_template

        I = load_image(input_file)
        res_y, res_x = I.shape[:2]
    else:
        res_x, res_y = resolution
    res_z = len(frames)

    max_res = max([res_x, res_y])

    shape = res_z, res_y, res_x

    fractal_noise_r = generate_3d_fractal_noise(shape, 0, 0, 0, scale, persistence, lacunarity, octaves)
    print("fractal_noise_r: ", np.min(fractal_noise_r), "-", np.max(fractal_noise_r))

    fractal_noise_t1 = generate_3d_fractal_noise(shape, shift_x1, shift_y1, shift_z1, scale, persistence, lacunarity,
                                                 octaves)
    print("fractal_noise_t1: ", np.min(fractal_noise_t1), "-", np.max(fractal_noise_t1))

    fractal_noise_t2 = generate_3d_fractal_noise(shape, shift_x2, shift_y2, shift_z2, scale, persistence, lacunarity,
                                                 octaves)
    print("fractal_noise_t2: ", np.min(fractal_noise_t2), "-", np.max(fractal_noise_t2))

    _ux_octave = fractal_noise_t1 - 0.5
    _uy_octave = fractal_noise_t2 - 0.5
    norm = np.sqrt(_ux_octave * _ux_octave + _uy_octave * _uy_octave) + 1.0e-5
    ux_octave = r_scale * fractal_noise_r * _ux_octave / norm
    uy_octave = r_scale * fractal_noise_r * _uy_octave / norm

    orig_x = np.arange(res_x)
    orig_y = np.arange(res_y)

    PX, PY = np.meshgrid(orig_x, orig_y)

    for i, frame in enumerate(tqdm.tqdm(frames)):
        try:
            input_file = input_file_template % frame
        except:
            input_file = input_file_template

        I = load_image(input_file)

        output_file = output_file_template % frame

        out_dir = os.path.dirname(output_file)
        os.makedirs(out_dir, exist_ok=True)

        SPX_OCT = np.clip((PX + max_res * ux_octave[i]).astype(int), 0, res_x - 1)
        SPY_OCT = np.clip((PY + max_res * uy_octave[i]).astype(int), 0, res_y - 1)

        if I.ndim == 3:
            O_oct = I[SPY_OCT, SPX_OCT, :]
        elif I.ndim == 2:
            O_oct = I[SPY_OCT, SPX_OCT]

        save_image(output_file, O_oct)

        save_image(output_file.replace(".png", ".hdf5"), np.flipud(O_oct))


def relocate_by_octave_noise_frame_image(
        I,
        frame=1,
        r_scale=0.01,
        scale_space=25,
        scale_time=0.01,
        persistence=5 / 10,
        lacunarity=20 / 10,
        octaves=7,
        shift_xyz=np.array([[0, 0, 0],
                            [48, 32, 40],
                            [175, 192, 180]]),
        repeat_xy=128,
        repeat_z=128,
        noise_resolution=(128, 128)
):
    """
    Warp an image using frame-dependent octave noise.

    Args:
        I (np.ndarray): Input image.
        frame (int): Frame index.
        r_scale, scale_space, scale_time, persistence, lacunarity, octaves (float/int): Noise parameters.
        shift_xyz (np.ndarray): 3x3 shift matrix.
        repeat_xy, repeat_z (int): Noise repeat intervals.
        noise_resolution (tuple): Resolution for noise generation.

    Returns:
        np.ndarray: Warped image.
    """
    shape = I.shape[:2]
    res_y, res_x = shape
    resolution = (res_x, res_y)
    max_res = max(res_x, res_y)

    fractal_noise_r = generate_fractal_noise_frame(frame, noise_resolution,
                                                   shift_xyz[0, 0], shift_xyz[0, 1], shift_xyz[0, 2],
                                                   scale_space, scale_time, persistence, lacunarity, octaves,
                                                   repeat_xy=repeat_xy,
                                                   repeat_z=repeat_z)
    # print("fractal_noise_r: ", np.min(fractal_noise_r), "-", np.max(fractal_noise_r))

    fractal_noise_t1 = generate_fractal_noise_frame(frame, noise_resolution,
                                                    shift_xyz[1, 0], shift_xyz[1, 1], shift_xyz[1, 2],
                                                    scale_space, scale_time, persistence, lacunarity,
                                                    octaves, repeat_xy=repeat_xy,
                                                    repeat_z=repeat_z)
    # print("fractal_noise_t1: ", np.min(fractal_noise_t1), "-", np.max(fractal_noise_t1))

    fractal_noise_t2 = generate_fractal_noise_frame(frame, noise_resolution,
                                                    shift_xyz[2, 0], shift_xyz[2, 1], shift_xyz[2, 2],
                                                    scale_space, scale_time,
                                                    persistence, lacunarity,
                                                    octaves, repeat_xy=repeat_xy,
                                                    repeat_z=repeat_z)
    # print("fractal_noise_t2: ", np.min(fractal_noise_t2), "-", np.max(fractal_noise_t2))

    _ux_octave = fractal_noise_t1 - 0.5
    _uy_octave = fractal_noise_t2 - 0.5
    norm = np.sqrt(_ux_octave * _ux_octave + _uy_octave * _uy_octave) + 1.0e-5
    ux_octave = r_scale * fractal_noise_r * _ux_octave / norm
    uy_octave = r_scale * fractal_noise_r * _uy_octave / norm

    ux_octave = cv2.resize(ux_octave, resolution)
    uy_octave = cv2.resize(uy_octave, resolution)

    orig_x = np.arange(res_x)
    orig_y = np.arange(res_y)

    PX, PY = np.meshgrid(orig_x, orig_y)

    SPX_OCT = np.clip((PX + max_res * ux_octave).astype(int), 0, res_x - 1)
    SPY_OCT = np.clip((PY + max_res * uy_octave).astype(int), 0, res_y - 1)

    if I.ndim == 3:
        O_oct = I[SPY_OCT, SPX_OCT, :]
    elif I.ndim == 2:
        O_oct = I[SPY_OCT, SPX_OCT]

    return O_oct


def relocate_by_octave_noise_frame(
        input_file, output_file,
        resolution=None,
        frame=1,
        r_scale=0.01,
        scale_space=25,
        scale_time=0.01,
        persistence=5 / 10,
        lacunarity=20 / 10,
        octaves=7,
        shift_x0=0,
        shift_y0=0,
        shift_z0=0,
        shift_x1=48,
        shift_y1=32,
        shift_z1=40,
        shift_x2=175,
        shift_y2=192,
        shift_z2=180,
        repeat=40,
        noise_resolution=(128, 128)):
    """
    Apply octave noise to a single RGBA image and save results.

    Args:
        input_file (str): Input image path.
        output_file (str): Output image path.
        resolution (tuple, optional): Target resolution.
        frame (int): Frame index for temporal noise.
        r_scale (float): Displacement scale.
        scale_space, scale_time (float): Frequency scales.
        persistence, lacunarity (float): Noise parameters.
        octaves (int): Number of noise octaves.
        shift_* (float): Noise vector offsets.
        repeat (int): Noise repetition cycle.
        noise_resolution (tuple): Resolution for noise generation.
    """
    I = load_rgba(input_file)
    if resolution is None:
        res_y, res_x = I.shape[:2]
        resolution = (res_x, res_y)
    else:
        res_x, res_y = resolution

    O = relocate_by_octave_noise_frame_image(
        I,
        frame=frame,
        r_scale=r_scale,
        scale_space=scale_space,
        scale_time=scale_time,
        persistence=persistence,
        lacunarity=lacunarity,
        octaves=octaves,
        noise_resolution=noise_resolution
    )
    save_rgba(output_file, O)

    save_hdf5(output_file.replace(".png", ".hdf5"), np.flipud(O))


def relocate_by_white_noise_file(input_file, output_file,
                                 resolution=None,
                                 r_scale=30 / 1000):
    """
    Apply random white noise to displace a single RGBA image.

    Args:
        input_file (str): Path to input RGBA image.
        output_file (str): Path to save warped image.
        resolution (tuple, optional): Image resolution.
        r_scale (float): Displacement scale factor.
    """
    I = load_rgba(input_file)

    if resolution is None:
        res_y, res_x = I.shape[:2]
        resolution = (res_x, res_y)
    else:
        res_x, res_y = resolution

    max_res = max([res_x, res_y])

    x1 = (np.arange(res_x) + 0.5) / max_res
    x2 = (np.arange(res_y) + 0.5) / max_res

    X1, X2 = np.meshgrid(x1, x2)

    shape = (res_y, res_x)

    orig_x = np.arange(res_x)
    orig_y = np.arange(res_y)

    PX, PY = np.meshgrid(orig_x, orig_y)

    rng = np.random.default_rng()
    white_r = rng.random(shape)
    white_t = rng.random(shape)

    ux_white = r_scale * white_r * np.cos(white_t * 2.0 * np.pi)
    uy_white = r_scale * white_r * np.sin(white_t * 2.0 * np.pi)

    SPX_WHT = np.clip((PX + max_res * ux_white).astype(int), 0, res_x - 1)
    SPY_WHT = np.clip((PY + max_res * uy_white).astype(int), 0, res_y - 1)
    O_wht = I[SPY_WHT, SPX_WHT, :]
    save_rgba(output_file, O_wht)

    save_hdf5(output_file.replace(".png", ".hdf5"), np.flipud(O_wht))


def relocate_by_white_noise_frames(input_file_template, output_file_template,
                                   frames=[1],
                                   resolution=None,
                                   r_scale=30 / 1000):
    """
    Apply white noise displacement to multiple RGBA image frames.

    Args:
        input_file_template (str): Template path for input images.
        output_file_template (str): Template path for output images.
        frames (List[int]): Frame indices.
        resolution (tuple, optional): Target resolution.
        r_scale (float): Displacement scale.
    """
    for frame in frames:
        input_file = input_file_template % frame
        output_file = output_file_template % frame

        relocate_by_white_noise_file(input_file, output_file,
                                     resolution=resolution,
                                     r_scale=r_scale)


class OctaveNoiseRelocator:
    """
    Applies 2D displacement to images using fractal octave noise.

    Attributes:
        r_scale (float): Displacement scale.
        scale_space, scale_time (float): Spatial/temporal noise scales.
        persistence, lacunarity (float): Noise structure parameters.
        octaves (int): Number of octaves for Perlin noise.
        shift_xyz (np.ndarray): 3D offsets for noise vector generation.
        repeat_xy, repeat_z (int): Repetition intervals.
        noise_resolution (tuple): Resolution used for noise sampling.
    """
    def __init__(self, r_scale=0.01,
                 scale_space=50,
                 scale_time=50,
                 persistence=5 / 10,
                 lacunarity=0.1,  # [0.1 - 2.0]...smaller value -> more clustered noise
                 octaves=7,
                 shift_xyz=[[0, 0, 0],
                            [48, 32, 40],
                            [175, 192, 180]],
                 repeat_xy=128,
                 repeat_z=128,
                 noise_resolution=(128, 128)):
        # print(f"shift_xyz: {shift_xyz}")
        # print(f"scale_space: {scale_space}")
        # print(f"scale_time: {scale_time}")
        self.r_scale = r_scale
        self.scale_space = scale_space
        self.scale_time = scale_time

        self.persistence = persistence

        self.lacunarity = lacunarity
        self.octaves = octaves
        self.shift_xyz = shift_xyz
        self.repeat_xy = repeat_xy
        self.repeat_z = repeat_z
        self.noise_resolution = noise_resolution

    def compute_noise(self,
                      frame,
                      r_scale):
        """
        Compute noise-based displacement fields for a given frame.

        Args:
            frame (int): Frame index.
            r_scale (float): Displacement strength.
        """

        noise_resolution = self.noise_resolution
        shift_xyz = self.shift_xyz
        scale_space = self.scale_space
        scale_time = self.scale_time
        persistence = self.persistence
        lacunarity = self.lacunarity
        octaves = self.octaves
        repeat_xy = self.repeat_xy
        repeat_z = self.repeat_z

        fractal_noise_r = generate_fractal_noise_frame(frame, noise_resolution,
                                                       shift_xyz[0, 0], shift_xyz[0, 1], shift_xyz[0, 2],
                                                       scale_space, scale_time, persistence, lacunarity, octaves,
                                                       repeat_xy=repeat_xy,
                                                       repeat_z=repeat_z)
        # print("fractal_noise_r: ", np.min(fractal_noise_r), "-", np.max(fractal_noise_r))

        fractal_noise_t1 = generate_fractal_noise_frame(frame, noise_resolution,
                                                        shift_xyz[1, 0], shift_xyz[1, 1], shift_xyz[1, 2],
                                                        scale_space, scale_time, persistence, lacunarity,
                                                        octaves, repeat_xy=repeat_xy,
                                                        repeat_z=repeat_z)
        # print("fractal_noise_t1: ", np.min(fractal_noise_t1), "-", np.max(fractal_noise_t1))

        fractal_noise_t2 = generate_fractal_noise_frame(frame, noise_resolution,
                                                        shift_xyz[2, 0], shift_xyz[2, 1], shift_xyz[2, 2],
                                                        scale_space, scale_time,
                                                        persistence, lacunarity,
                                                        octaves, repeat_xy=repeat_xy,
                                                        repeat_z=repeat_z)
        # print("fractal_noise_t2: ", np.min(fractal_noise_t2), "-", np.max(fractal_noise_t2))

        _ux_octave = fractal_noise_t1 - 0.5
        _uy_octave = fractal_noise_t2 - 0.5
        norm = np.sqrt(_ux_octave * _ux_octave + _uy_octave * _uy_octave) + 1.0e-5
        ux_octave = r_scale * fractal_noise_r * _ux_octave / norm
        uy_octave = r_scale * fractal_noise_r * _uy_octave / norm

        self.ux_octave = ux_octave
        self.uy_octave = uy_octave

    def relocate(self, I, frame, r_scale=0.02):
        """
        Apply precomputed displacement field to image.

        Args:
            I (np.ndarray): Input image.
            frame (int): Frame index.
            r_scale (float): Displacement scale.

        Returns:
            np.ndarray: Warped image.
        """
        self.compute_noise(frame, r_scale)

        ux_octave = self.ux_octave
        uy_octave = self.uy_octave

        res_y, res_x = I.shape[:2]

        resolution = (res_x, res_y)

        ux_octave = cv2.resize(ux_octave, resolution)
        uy_octave = cv2.resize(uy_octave, resolution)

        res_y, res_x = I.shape[:2]
        orig_x = np.arange(res_x)
        orig_y = np.arange(res_y)

        max_res = max(res_x, res_y)

        PX, PY = np.meshgrid(orig_x, orig_y)

        SPX_OCT = np.clip((PX + max_res * ux_octave).astype(int), 0, res_x - 1)
        SPY_OCT = np.clip((PY + max_res * uy_octave).astype(int), 0, res_y - 1)

        if I.ndim == 3:
            O_oct = I[SPY_OCT, SPX_OCT, :]
        elif I.ndim == 2:
            O_oct = I[SPY_OCT, SPX_OCT]

        return O_oct
