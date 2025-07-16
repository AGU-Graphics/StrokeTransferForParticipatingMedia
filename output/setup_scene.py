# -----------------------------------------------------------------------------
# Stroke Transfer for Participating Media
# http://cg.it.aoyama.ac.jp/yonghao/sig25/abstsig25.html
#
# File: output/setup_scene.py
# Maintainer: Hideki Todo
#
# Description:
# Setup script for initializing a scene in the output directory by:
# - Creating a symbolic link to the assets directory.
# - Automatically cloning the required volume data from a remote repository.
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
import shutil
import json


def load_volume_repo_info(json_path="../volume_data_repositories.json"):
    """
        Load the scene-to-repo mapping from a JSON file.
        """
    if not os.path.exists(json_path):
        raise FileNotFoundError(f"[ERROR] Repository info file not found: {json_path}")
    with open(json_path, "r") as f:
        return json.load(f)


def confirm_and_clone(scene_name, repo_info):
    """
    Confirm with the user and clone volume data for the given scene if not already present.

    Assumes the remote repository contains a 'volume_data/' directory at the top level.
    The contents of 'volume_data/' will be moved into: assets/[scene_name]/volume_data/
    """
    info = repo_info.get(scene_name)
    if info is None:
        print(f"[WARN] No volume repo information defined for scene: {scene_name}")
        return

    repo_url = info["repo_url"]
    size_gb = info.get("size_gb", "unknown")

    target_path = os.path.join("../assets", scene_name, "volume_data")
    if os.path.exists(target_path):
        print(f"[INFO] Volume data already exists at: {target_path}")
        return

    print(f"[INFO] Volume data for scene '{scene_name}' will be cloned from:")
    print(f"       {repo_url}")
    print(f"       (Expected size: {size_gb} GB)")
    answer = input("Proceed with clone? [y/N]: ").strip().lower()

    if answer not in ("y", "yes"):
        print("[INFO] Clone cancelled by user.")
        return

    temp_clone_path = os.path.join("../assets", scene_name, "_volume_repo_temp")
    os.makedirs(os.path.dirname(temp_clone_path), exist_ok=True)

    print(f"[INFO] Cloning repository into temporary location: {temp_clone_path}")
    subprocess.run(["git", "clone", "--depth", "1", repo_url, temp_clone_path], check=True)

    src_volume_data = os.path.join(temp_clone_path, "volume_data")
    if not os.path.exists(src_volume_data):
        print(f"[ERROR] 'volume_data/' directory not found in cloned repository: {src_volume_data}")
        shutil.rmtree(temp_clone_path)
        return

    print(f"[INFO] Moving volume_data/ to final location: {target_path}")
    shutil.move(src_volume_data, target_path)

    print(f"[INFO] Cleaning up temporary clone directory: {temp_clone_path}")
    shutil.rmtree(temp_clone_path)

    print(f"[INFO] Volume data setup complete: {target_path}")


def link_assets(scene_name):
    """
    Create a symbolic link: output/[scene_name]/assets -> ../assets/[scene_name]
    """
    src = os.path.abspath(os.path.join("../assets", scene_name))
    dst_dir = scene_name
    dst = os.path.join(dst_dir, "assets")

    os.makedirs(dst_dir, exist_ok=True)

    if os.path.exists(dst):
        if os.path.islink(dst):
            print(f"[INFO] Symlink already exists: {dst}")
        else:
            raise RuntimeError(f"[ERROR] {dst} exists and is not a symlink.")
    else:
        os.symlink(src, dst)
        print(f"[INFO] Symlink created: {dst} -> {src}")


def main():
    parser = argparse.ArgumentParser(description="Set up volume data and symbolic link for scene.")
    parser.add_argument("scene_name", help="Name of the scene to set up.")
    args = parser.parse_args()

    link_assets(args.scene_name)
    repo_info = load_volume_repo_info()
    confirm_and_clone(args.scene_name, repo_info)


if __name__ == "__main__":
    main()
