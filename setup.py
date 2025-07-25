# -----------------------------------------------------------------------------
# Stroke Transfer for Participating Media
# http://cg.it.aoyama.ac.jp/yonghao/sig25/abstsig25.html
#
# File: setup.py
# Maintainer: Yonghao Yue and Hideki Todo
#
# Description:
# Setup script for building C++ tools and configuring the development environment using conda or pip
# for the main pipelines of "Stroke Transfer for Participating Media".
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
import argparse
import subprocess
import sys

BUILD_DIR = "./code/cpp_tools/build"
BIN_DIR = "./code/cpp_tools/bin"


def create_symlink(proj):
    src = os.path.abspath(os.path.join(BUILD_DIR, proj, proj))
    dst = os.path.join(BIN_DIR, proj)

    try:
        if os.path.islink(dst) or os.path.exists(dst):
            os.remove(dst)
        os.symlink(src, dst)
        print(f"[INFO] Symlink created: {dst} -> {src}")
    except OSError as e:
        print(f"[WARN] Could not create symlink for {proj}: {e}")


def build_cpp_tools():
    print("[INFO] Building C++ tools...")
    os.makedirs(BUILD_DIR, exist_ok=True)
    os.makedirs(BIN_DIR, exist_ok=True)

    current_dir = os.getcwd()
    os.chdir(BUILD_DIR)
    os.system('cmake -DCMAKE_POLICY_VERSION_MINIMUM=3.5 ..')

    cpu_count = os.cpu_count()
    os.system(f'make -j{cpu_count}')

    os.chdir(current_dir)

    projs = [
        "exr2hdf_cli",
        "gen_strokes_cli",
        "oil_lighting_editor_gui",
        "pencil_lighting_editor_gui",
        "render_strokes_cli",
        "render_strokes_pencil_cli",
        "render_surface_features_cli"
    ]

    for p in projs:
        create_symlink(p)


def run_command(cmd, cwd=None):
    print(f"[Running] {cmd}")
    result = subprocess.run(cmd, shell=True, cwd=cwd)
    if result.returncode != 0:
        print(f"[ERROR] Command failed: {cmd}")
        sys.exit(1)


def setup_conda_env():
    print("[INFO] Setting up environment via conda...")
    run_command("conda env create -f environment.yml")


def setup_pip_env():
    print("[INFO] Setting up environment via pip...")
    run_command("pip install -U pip")
    run_command("pip install -r requirements.txt")


def parse_args():
    parser = argparse.ArgumentParser(description="Setup stroke_transfer environment.")
    parser.add_argument("--env", type=str, choices=["conda", "pip", "cpp"], required=True,
                        help="Environment type: 'conda', 'pip', or 'cpp'")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()

    if args.env == "cpp":
        build_cpp_tools()
    else:
        build_cpp_tools()
        if args.env == "conda":
            setup_conda_env()
        elif args.env == "pip":
            setup_pip_env()

    print("[INFO] Setup complete.")
