#!/usr/bin/env python3
import cv2
import numpy as np
from multiprocessing import Pool
from tqdm import tqdm
import subprocess
import argparse

from generate_image import ImageGenerator


def helper(params):
    width, height, angle = params
    ig = ImageGenerator(width, height)
    return ig.generate(angle)


def main():
    parser = argparse.ArgumentParser(description='Generate a rotating globe video.')
    parser.add_argument("width", help="width of output video", type=int)
    parser.add_argument("height", help="height of output video", type=int)
    args = parser.parse_args()

    image_width = args.width
    image_height = args.height

    filepath = f"media/video_{image_width}x{image_height}.mp4"
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    fps = 60
    writer = cv2.VideoWriter(filepath, fourcc, fps, (image_width, image_height))

    angles = np.arange(0, 360, 0.5)
    params = ((image_width, image_height, angle) for angle in angles)

    with Pool() as pool:
        for frame in tqdm(pool.imap(helper, params), total=len(angles)):
            writer.write(frame)
    writer.release()

    temppath = "media/temp.mp4"
    subprocess.run(f"mv {filepath} {temppath}".split(" "))
    subprocess.run(f"ffmpeg -an -i {temppath} -vcodec libx264 -pix_fmt yuv420p -profile:v baseline -level 3 {filepath}".split(" "))
    subprocess.run(f"rm {temppath}".split(" "))
    # subprocess.run(f"mpv {filepath} --fs --loop".split(" "))


if __name__ == '__main__':
    main()
