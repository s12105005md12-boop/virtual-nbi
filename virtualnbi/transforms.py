# -*- coding: utf-8 -*-
"""
virtualnbi.transforms
=====================
Spectral transforms that convert white-light (WLI) endoscopic RGB frames into
NBI-like virtual narrow-band imaging (vNBI) channels.

Core concept
------------
NBI's diagnostic power comes from narrow-band illumination at ~415 nm (blue;
superficial mucosal microvasculature) and ~540 nm (green; deeper vessels),
which enhances hemoglobin contrast. A WLI RGB frame captured under broad
spectrum illumination contains overlapping spectral information; vNBI
re-weights and combines the RGB channels to approximate NBI-like contrast.

Methods (configurable via `method`)
-----------------------------------
- 'channel'   : band re-weighting  NBI_est = a*B + b*G - c*R (clamped)
- 'log_ratio' : hemoglobin-contrast maps  log(G/B), log(R/G), log(G*B/R)
- 'matrix'    : fixed 3x3 spectral color matrix (i-Scan/FICE-like)
- 'ica'       : independent-component spectral decomposition (optional)
- 'frangi'    : Frangi vesselness computed on green / virtual-NBI channel

All methods are pure NumPy; scikit-image is optional (used only for Frangi).
`VirtualNBI.transform()` returns N channels concatenated.

Reference
---------
Lin J-H, Chen L-F, Yang M-H. AI-Based Conversion of White-Light Endoscopy to
Narrow-Band Imaging: Virtual NBI with Open-Source, Real-Time Code (2026).
"""
import numpy as np

__all__ = [
    "VirtualNBI",
    "channel_combination",
    "log_ratio_maps",
    "spectral_matrix",
    "ica_channels",
    "frangi_vesselness",
]


def _to_float01(rgb):
    rgb = np.asarray(rgb, dtype=np.float32)
    if rgb.max() > 1.0 + 1e-6:
        rgb = rgb / 255.0
    return rgb


def channel_combination(rgb, a=0.9, b=0.6, c=0.35):
    """NBI-like single channel: emphasize B/G, suppress R (hemoglobin darkening)."""
    r, g, b = rgb[..., 0], rgb[..., 1], rgb[..., 2]
    out = a * b + b * g - c * r
    return np.clip(out, 0.0, 1.0)


def log_ratio_maps(rgb, eps=1e-6):
    """Hemoglobin-contrast maps (modified Beer-Lambert style)."""
    r, g, b = rgb[..., 0] + eps, rgb[..., 1] + eps, rgb[..., 2] + eps
    gb = np.log(g / b)  # 'hemoglobin concentration' proxy
    rg = np.log(r / g)  # 'oxygenation' proxy
    gbr = np.log(g * b / (r + eps))
    return np.stack([gb, rg, gbr], axis=-1)


def spectral_matrix(rgb):
    """Fixed 3x3 spectral re-weighting (i-Scan/FICE-like colour matrix)."""
    M = np.array([[0.0, 0.8, 0.9],
                  [0.5, 0.6, 0.2],
                  [0.7, 0.3, 0.0]], dtype=np.float32)
    out = rgb @ M.T
    return np.clip(out, 0.0, 1.0)


def ica_channels(rgb, n_components=3, max_iter=500):
    """ICA spectral decomposition (optional; needs scikit-learn)."""
    from sklearn.decomposition import FastICA
    h, w, c = rgb.shape
    flat = rgb.reshape(-1, c)
    ica = FastICA(n_components=min(n_components, c), max_iter=max_iter, random_state=0)
    S = ica.fit_transform(flat)
    S = (S - S.min(axis=0, keepdims=True)) / (np.ptp(S, axis=0, keepdims=True) + 1e-9)
    return S.reshape(h, w, n_components)


def frangi_vesselness(rgb, channel="green"):
    """Frangi vesselness (optional; needs scikit-image). Scale in [0,1]."""
    from skimage.filters import frangi
    if channel == "green":
        src = rgb[..., 1]
    else:
        src = channel_combination(rgb)
    v = frangi(src, sigmas=range(1, 5, 1), black_ridges=False)
    return np.clip((v - v.min()) / (np.ptp(v) + 1e-9), 0.0, 1.0)


class VirtualNBI:
    """Configurable virtual-NBI spectral analysis transform."""

    def __init__(self, method="matrix", n_out=3):
        self.method = method
        self.n_out = n_out

    def transform(self, rgb):
        rgb = _to_float01(rgb)
        if self.method == "channel":
            out = channel_combination(rgb)
            return np.stack([out] * self.n_out, axis=-1)
        if self.method == "log_ratio":
            return log_ratio_maps(rgb)[..., : self.n_out]
        if self.method == "matrix":
            return spectral_matrix(rgb)[..., : self.n_out]
        if self.method == "ica":
            return ica_channels(rgb, n_components=self.n_out)
        if self.method == "frangi":
            out = frangi_vesselness(rgb)
            return np.stack([out] * self.n_out, axis=-1)
        raise ValueError("unknown method: %s" % self.method)

    def __call__(self, rgb):
        return self.transform(rgb)
