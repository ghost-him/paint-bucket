"""sRGB <-> CIELab (D65) conversion for float32 image arrays.

All pipeline maths happens in Lab: it is perceptually uniform, so a single
distance threshold ("1.5 levels") means the same thing in shadows and highlights.
"""

from __future__ import annotations

import numpy as np

# sRGB (IEC 61966-2-1) primaries, D65 white point
_M_RGB2XYZ = np.array(
    [
        [0.4124564, 0.3575761, 0.1804375],
        [0.2126729, 0.7151522, 0.0721750],
        [0.0193339, 0.1191920, 0.9503041],
    ],
    dtype=np.float32,
)
_M_XYZ2RGB = np.array(
    [
        [3.2404542, -1.5371385, -0.4985314],
        [-0.9692660, 1.8760108, 0.0415560],
        [0.0556434, -0.2040259, 1.0572252],
    ],
    dtype=np.float32,
)
_WHITE = np.array([0.95047, 1.0, 1.08883], dtype=np.float32)
_DELTA = 6.0 / 29.0


def srgb_to_lab(rgb01: np.ndarray) -> np.ndarray:
    """rgb01: (..., 3) float in 0..1 -> Lab float32 (L 0..100, a/b roughly -128..127)."""
    rgb01 = np.asarray(rgb01, dtype=np.float32)
    lin = np.where(rgb01 <= 0.04045, rgb01 / 12.92, ((rgb01 + 0.055) / 1.055) ** 2.4)
    xyz = (lin @ _M_RGB2XYZ.T) / _WHITE
    f = np.where(xyz > _DELTA**3, np.cbrt(xyz), xyz / (3 * _DELTA**2) + 4.0 / 29.0)
    return np.stack(
        [116.0 * f[..., 1] - 16.0, 500.0 * (f[..., 0] - f[..., 1]), 200.0 * (f[..., 1] - f[..., 2])],
        axis=-1,
    ).astype(np.float32)


def lab_to_srgb(lab: np.ndarray) -> np.ndarray:
    """Lab -> (..., 3) float in 0..1, clipped to the sRGB gamut."""
    lab = np.asarray(lab, dtype=np.float32)
    fy = (lab[..., 0] + 16.0) / 116.0
    fx = fy + lab[..., 1] / 500.0
    fz = fy - lab[..., 2] / 200.0
    f = np.stack([fx, fy, fz], axis=-1)
    xyz = np.where(f > _DELTA, f**3, 3 * _DELTA**2 * (f - 4.0 / 29.0)) * _WHITE
    lin = xyz @ _M_XYZ2RGB.T
    lin = np.clip(lin, 0.0, 1.0)
    return np.where(lin <= 0.0031308, lin * 12.92, 1.055 * lin ** (1.0 / 2.4) - 0.055).astype(np.float32)


def lab_delta(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Euclidean Lab distance (dE76) between (..., 3) arrays."""
    d = np.asarray(a, np.float32) - np.asarray(b, np.float32)
    return np.sqrt((d * d).sum(-1))


def luma(rgb: np.ndarray) -> np.ndarray:
    """Rec.709 luma from (..., 3) rgb in any common scale."""
    rgb = np.asarray(rgb, dtype=np.float32)
    return 0.2126 * rgb[..., 0] + 0.7152 * rgb[..., 1] + 0.0722 * rgb[..., 2]
