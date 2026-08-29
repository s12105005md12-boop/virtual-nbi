#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Example: convert a white-light endoscopy image (or a folder of frames) into
virtual narrow-band imaging (vNBI), and save a side-by-side comparison.

Usage
-----
    python demo_convert.py path/to/wli_image.jpg                # single image
    python demo_convert.py path/to/frames/ --out out_dir/       # folder of frames
    python demo_convert.py video.mp4 --out out.gif --video       # video -> GIF

Outputs
-------
  single image : out_dir/compare.png  (left: WLI | right: vNBI)
  folder       : out_dir/compare_*.png for each input frame
  video        : out_dir/<name>.gif  (frame-synchronized WLI | vNBI)
"""
import argparse
import glob
import os
import sys

import numpy as np
from PIL import Image, ImageDraw

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from virtualnbi import VirtualNBI  # noqa: E402


def convert_one(rgb, method):
    vn = VirtualNBI(method=method, n_out=3)
    v = np.clip(vn(rgb), 0, 1)
    side = np.concatenate([rgb, v], axis=1)
    return np.asarray(side * 255, dtype=np.uint8)


def draw_labels(panel):
    im = Image.fromarray(panel)
    d = ImageDraw.Draw(im)
    d.rectangle([0, 0, im.width, 18], fill=(0, 0, 0))
    d.text((10, 3), "WLI", fill=(255, 255, 255))
    d.text((im.width // 2 + 10, 3), "vNBI", fill=(255, 255, 255))
    return im


def main():
    ap = argparse.ArgumentParser(description="WLI -> virtual NBI conversion demo")
    ap.add_argument("input", help="image file, folder of frames, or video file")
    ap.add_argument("--out", default="output", help="output directory or GIF path")
    ap.add_argument("--method", default="matrix",
                    choices=["matrix", "channel", "log_ratio", "ica", "frangi"])
    ap.add_argument("--video", action="store_true",
                    help="input is a video file; write a synchronized GIF")
    args = ap.parse_args()

    os.makedirs(args.out, exist_ok=True) if not args.video else None

    if args.video:
        # frame extraction via ffmpeg
        import subprocess
        import tempfile

        tmp = tempfile.mkdtemp(prefix="vnbi_frames_")
        subprocess.run(
            ["ffmpeg", "-y", "-loglevel", "error", "-i", args.input,
             "-vf", "fps=5,scale=640:-1", os.path.join(tmp, "frame_%03d.jpg")],
            check=True,
        )
        frames = sorted(glob.glob(os.path.join(tmp, "*.jpg")))
        panels = []
        for fp in frames:
            rgb = np.asarray(Image.open(fp).convert("RGB"), dtype=np.float32) / 255.0
            panels.append(draw_labels(convert_one(rgb, args.method)))
        gif = args.out if args.out.endswith(".gif") else os.path.join(args.out, "realtime.gif")
        panels[0].save(gif, save_all=True, append_images=panels[1:], duration=200, loop=0)
        print("saved synchronized GIF:", gif, "| frames:", len(panels))
        return

    if os.path.isdir(args.input):
        files = sorted(glob.glob(os.path.join(args.input, "*")))
        imgs = [f for f in files if f.lower().endswith((".png", ".jpg", ".jpeg"))]
        for fp in imgs:
            rgb = np.asarray(Image.open(fp).convert("RGB"), dtype=np.float32) / 255.0
            name = os.path.splitext(os.path.basename(fp))[0]
            panel = draw_labels(convert_one(rgb, args.method))
            panel.save(os.path.join(args.out, f"compare_{name}.png"))
        print("processed", len(imgs), "frames ->", args.out)
        return

    rgb = np.asarray(Image.open(args.input).convert("RGB"), dtype=np.float32) / 255.0
    panel = draw_labels(convert_one(rgb, args.method))
    out = os.path.join(args.out, "compare.png")
    panel.save(out)
    print("saved:", out, "| input shape:", rgb.shape, "| method:", args.method)


if __name__ == "__main__":
    main()
