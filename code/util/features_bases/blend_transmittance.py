# -----------------------------------------------------------------------------
# Stroke Transfer for Participating Media
# http://cg.it.aoyama.ac.jp/yonghao/sig25/abstsig25.html
#
# File: blend_transmittance.py
# Maintainer: Naoto Shirashima
#
# Description:
# Integrates the transmittance of medium and surface models
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
import sys
from data_io import DataIO2D, Output2D

def main():
    media_transmittance_path = sys.argv[1]
    surface_transmittance_path = sys.argv[2]
    output_path = sys.argv[3]

    surface_transmittance_data = DataIO2D.InitLoadFile(surface_transmittance_path, taichi=False).data_np
    media_transmittance_data = DataIO2D.InitLoadFile(media_transmittance_path, taichi=False).data_np
    mix_transmittance_data = np.minimum(surface_transmittance_data, media_transmittance_data)

    Output2D(mix_transmittance_data, output_path)


if __name__ == '__main__':
    main()