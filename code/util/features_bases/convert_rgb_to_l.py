# -----------------------------------------------------------------------------
# Stroke Transfer for Participating Media
# http://cg.it.aoyama.ac.jp/yonghao/sig25/abstsig25.html
#
# File: convert_rgb_to_l.py
# Maintainer: Naoto Shirashima and Yuki Yamaoka
#
# Description:
# Conversion from RGB to L* in Lab color space
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
import os
import numpy as np
import sys
import h5py
import cv2
from data_io import DataIO2D, Output2D

def rgb_to_xyz(rgb):
    h, w = rgb.shape[:2]
    rgb = rgb.reshape(-1, 3)

    rgb_linear = np.where(rgb > 0.04045, ((rgb + 0.055) / 1.055) ** 2.4, rgb / 12.92)

    M = np.array([[0.4124564, 0.3575761, 0.1804375],
                  [0.2126729, 0.7151522, 0.0721750],
                  [0.0193339, 0.1191920, 0.9503041]])
    xyz = np.dot(rgb_linear, M.T)
    xyz = xyz.reshape(h, w, -1)
    return xyz

def xyz_to_lab(xyz):
    h, w = xyz.shape[:2]
    xyz = xyz.reshape(-1, 3)
    ref_white = np.array([0.95047, 1.0, 1.08883])

    xyz_normalized = np.array(xyz)

    for ci in range(3):
        xyz_normalized[:, ci] = xyz[:, ci] / ref_white[ci]
    epsilon = 0.008856
    kappa = 903.3

    mask = xyz_normalized > epsilon
    xyz_f = np.where(mask, xyz_normalized ** (1 / 3), (kappa * xyz_normalized + 16) / 116)

    L = 116 * xyz_f[:, 1] - 16
    a = 500 * (xyz_f[:, 0] - xyz_f[:, 1])
    b = 200 * (xyz_f[:, 1] - xyz_f[:, 2])

    lab = np.vstack((L, a, b)).T
    lab = lab.reshape(h, w, -1)
    return lab

def rgb_to_lab(I):
    xyz = rgb_to_xyz(I)
    lab = xyz_to_lab(xyz)

    return lab


def main():
    input_r_path = sys.argv[1]
    input_g_path = sys.argv[2]
    input_b_path = sys.argv[3]
    output_path = sys.argv[4]

    in_r = DataIO2D.InitLoadFile(input_r_path, taichi=False).data_np
    in_g = DataIO2D.InitLoadFile(input_g_path, taichi=False).data_np
    in_b = DataIO2D.InitLoadFile(input_b_path, taichi=False).data_np

    RGB = np.zeros((in_r.shape[0], in_r.shape[1], 3))
    RGB[:,:,0] = in_r
    RGB[:,:,1] = in_g
    RGB[:,:,2] = in_b
    Lab = rgb_to_lab(RGB)
    L = Lab[:,:,0]
    Output2D( L, output_path )

if __name__ == '__main__':
    main()