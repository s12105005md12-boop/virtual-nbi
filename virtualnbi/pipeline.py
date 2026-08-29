# -*- coding: utf-8 -*-
"""
virtualnbi.pipeline
===================
Frame-synchronized WLI -> vNBI pipeline for live endoscopy or archived video.

Wraps `virtualnbi.transforms.VirtualNBI` into a per-frame transform and
provides a torch-friendly tensor builder (RGB + vNBI channels) for downstream
deep-learning consumption.

Experiment variants supported:
  E8a  WLI only (baseline, RGB 3ch)
  E8b  WLI + virtual-NBI channel (RGB 4ch, or parallel branch)
  E8c  WLI + vesselness-prior -> attention bias
"""
import numpy as np

from .transforms import VirtualNBI, frangi_vesselness

__all__ = ["VirtualNBIPipeline", "convert_image", "convert_video_frames"]


class VirtualNBIPipeline:
    def __init__(self, method="matrix", n_out=3, with_vesselness=False):
        self.transform = VirtualNBI(method=method, n_out=n_out)
        self.with_vesselness = with_vesselness

    def __call__(self, rgb):
        """rgb: (H,W,3) uint8/float. Returns dict of numpy arrays."""
        vnbi = self.transform(rgb)
        out = {"virtual_nbi": vnbi}
        if self.with_vesselness:
            out["vesselness"] = frangi_vesselness(rgb)
        return out

    def to_tensor(self, rgb):
        """Stack WLI RGB + virtual-NBI channels into (4+..., H, W) float tensor.

        Returns (tensor, dict_of_aux) where aux holds vesselness for attention.
        """
        import torch

        rgb = np.asarray(rgb, dtype=np.float32)
        if rgb.max() > 1.0:
            rgb = rgb / 255.0
        vnbi = self.transform(rgb)
        x = torch.from_numpy(np.concatenate([rgb, vnbi], axis=-1).transpose(2, 0, 1))
        aux = {}
        if self.with_vesselness:
            aux["vesselness"] = torch.from_numpy(frangi_vesselness(rgb))[None]
        return x, aux


def convert_image(rgb, method="matrix"):
    """Convert a single WLI RGB frame (H,W,3) to vNBI display output (H,W,3)."""
    return VirtualNBI(method=method, n_out=3)(rgb)


def convert_video_frames(frames, method="matrix", side_by_side=True):
    """Convert an iterable of WLI frames into (a list of) vNBI frames.

    If `side_by_side` is True, each output frame concatenates [WLI | vNBI]
    horizontally (uint8, ready to be written to a video/GIF).
    """
    vn = VirtualNBI(method=method, n_out=3)
    out = []
    for rgb in frames:
        rgb = np.asarray(rgb, dtype=np.float32)
        if rgb.max() > 1.0:
            rgb = rgb / 255.0
        v = np.clip(vn(rgb), 0, 1)
        if side_by_side:
            panel = np.concatenate([rgb, v], axis=1)
            out.append(np.asarray(panel * 255, dtype=np.uint8))
        else:
            out.append(np.asarray(v * 255, dtype=np.uint8))
    return out
