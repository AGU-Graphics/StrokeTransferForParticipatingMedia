# -----------------------------------------------------------------------------
# Stroke Transfer for Participating Media
# http://cg.it.aoyama.ac.jp/yonghao/sig25/abstsig25.html
#
# File: util/image/image2video.py
# Maintainer: Hideki Todo
#
# Description:
# Utility module for converting a sequence of images into a video using FFmpeg or OpenCV.
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
import shutil
import subprocess
import tempfile

import cv2

# VCODEC = 'libx264'
VCODEC = None


def images_to_video(input_template, out_file, frame_rate, frames, use_ffmpeg=True):
    """Convert a sequence of images into a video using ffmpeg or OpenCV.

        Args:
            input_template (str): Template string for input image files (e.g., "frame_%03d.png").
            out_file (str): Path to the output video file.
            frame_rate (float): Frames per second of the output video.
            frames (List[int]): List of frame indices to include.
            use_ffmpeg (bool): Whether to use ffmpeg (if False, uses OpenCV).
        """
    if use_ffmpeg:
        images_to_video_ffmpeg(input_template, out_file, frame_rate, frames)
    else:
        images_to_video_cv(input_template, out_file, frame_rate, frames)


def run_ffmpeg_command(input_template, out_file, frame_rate=8):
    """Run ffmpeg command to convert images into a video.

        Args:
            input_template (str): Template string for temporary image files (e.g., "image_%d.png").
            out_file (str): Output video file path.
            frame_rate (float, optional): Frame rate of the output video. Defaults to 8.
        """
    cmd = f'ffmpeg -framerate {frame_rate} -i {input_template}'
    if VCODEC is not None:
        cmd += f' -vcodec {VCODEC}'
    cmd += f' -pix_fmt yuv420p -qmax 16 -vf scale=trunc(iw/2)*2:trunc(ih/2)*2 -y -loglevel error {out_file}'
    subprocess.call(cmd.split(' '))


def images_to_video_ffmpeg(input_template, out_file, frame_rate, frames):
    """Convert a sequence of images into a video using ffmpeg backend.

        Args:
            input_template (str): Template for image file paths (e.g., "frame_%03d.png").
            out_file (str): Output video file path.
            frame_rate (float): Desired frame rate.
            frames (List[int]): List of frame indices to include.
        """
    if os.path.exists(out_file):
        os.remove(out_file)

    with tempfile.TemporaryDirectory() as temp_dir:
        temp_file_template = os.path.join(temp_dir, "image_%d.png")
        frame_skip = frames[1] - frames[0]
        for i, frame in enumerate(frames):
            image_file = input_template % frame

            if not os.path.exists(image_file):
                break

            temp_file = temp_file_template % i
            shutil.copy2(image_file, temp_file)

        out_dir = os.path.dirname(out_file)
        os.makedirs(out_dir, exist_ok=True)
        run_ffmpeg_command(temp_file_template, out_file, frame_rate=frame_rate / frame_skip)


def images_to_video_cv(input_template, out_file, frame_rate, frames):
    """Convert a sequence of images into a video using OpenCV backend.

        Args:
            input_template (str): Template for image file paths (e.g., "frame_%03d.png").
            out_file (str): Output video file path.
            frame_rate (float): Desired frame rate.
            frames (List[int]): List of frame indices to include.
        """
    image_files = []

    for frame in frames:
        image_file = input_template % frame
        image_files.append(image_file)

    I = cv2.imread(image_files[0])
    h, w = I.shape[:2]
    size = (w, h)

    fourcc = cv2.VideoWriter_fourcc('m', 'p', '4', 'v')

    writer = cv2.VideoWriter(out_file, fourcc, frame_rate, size, True)

    for image_file in image_files:
        I = cv2.imread(image_file)
        writer.write(I)

    writer.release()
