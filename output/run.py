# -----------------------------------------------------------------------------
# Stroke Transfer for Participating Media
# http://cg.it.aoyama.ac.jp/yonghao/sig25/abstsig25.html
#
# File: output/run.py
# Maintainer: Hideki Todo
#
# Description:
# Command-line entry point for running stroke transfer pipelines.
# - Loads scene-specific settings from a JSON file.
# - Executes selected pipelines for each specified scene based on command-line options.
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
import sys
import argparse
import os

_output_dir = os.path.dirname(os.path.abspath(__file__))
_code_dir = os.path.abspath(os.path.join(_output_dir, "../code"))
sys.path.append(_code_dir)

from stroke_transfer import run_pipeline


def get_default_setting_file():
    return "assets/pipeline_settings.json"


def parse_args():
    parser = argparse.ArgumentParser(description="Run stroke transfer pipeline.")

    parser.add_argument("--settings", type=str, default=get_default_setting_file(),
                        help="Path to pipeline_settings.json")

    parser.add_argument("--frame_start", type=int, default=None, help="Override frame start")
    parser.add_argument("--frame_end", type=int, default=None, help="Override frame end")
    parser.add_argument("--frame_skip", type=int, default=None, help="Override frame skip")

    parser.add_argument("--scene_names", type=str, nargs='+', default=None,
                        help="List of scene names to process (e.g., Rising_smoke Ring_fire_dense)")
    parser.add_argument("--pipelines", type=str, nargs='+', default=None,
                        help="List of pipelines to run (e.g., p5_regression p7_transfer)")
    parser.add_argument("--transfer_targets", type=str, nargs='+', default=None,
                        help="List of transfer targets to run (orientation/color/width/length)")
    parser.add_argument("--verbose", action="store_true",
                        help="Enable verbose debug output")
    parser.add_argument("--plot", action="store_true",
                        help="Generate feature plot images")

    parser.add_argument("--list-pipelines", action="store_true",
                        help="List available pipeline names and exit")

    parser.add_argument("--list-pipeline-groups", action="store_true",
                        help="List available pipeline group names and exit")

    return parser.parse_args()


if __name__ == '__main__':
    args = parse_args()

    if args.list_pipelines:
        from stroke_transfer import list_available_pipelines

        list_available_pipelines()
        sys.exit(0)

    if args.list_pipeline_groups:
        from stroke_transfer import list_pipeline_groups

        list_pipeline_groups()
        sys.exit(0)

    frame_start = args.frame_start
    frame_end = args.frame_end
    frame_skip = args.frame_skip

    if args.frame_start is not None and args.frame_end is not None and args.frame_skip is not None:
        frames = list(range(args.frame_start, args.frame_end + 1, args.frame_skip))
    else:
        frames = None

    for scene_name in args.scene_names:
        scene_dir = os.path.join(_output_dir, scene_name)
        os.chdir(scene_dir)
        run_pipeline(args.settings,
                     frame_start=frame_start, frame_end=frame_end, frame_skip=frame_skip,
                     frames=frames, pipelines=args.pipelines,
                     verbose=args.verbose,
                     plot=args.plot,
                     transfer_targets=args.transfer_targets)
