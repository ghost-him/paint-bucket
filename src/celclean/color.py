"""sRGB <-> CIELab (D65) conversion for float32 image arrays.

All pipeline maths happens in Lab: it is perceptually uniform, so a single
distance threshold ("1.5 levels") means the same thing in shadows and highlights.
"""

from __future__ import annotations

import numpy as np

from .ops import row_bands, run_bands

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

# sRGB byte -> linear light. Exact for every one of the 256 byte values, so 8-bit input can take
# the table instead of the pow() (the pipeline always starts from a uint8 image).
_SRGB8 = np.arange(256, dtype=np.float32) / np.float32(255.0)
_SRGB8_TO_LINEAR = np.where(
    _SRGB8 <= 0.04045, _SRGB8 / 12.92, ((_SRGB8 + 0.055) / 1.055) ** 2.4
).astype(np.float32)


def _srgb_block_to_lab(rgb01: np.ndarray) -> np.ndarray:
    """One row band of `srgb_to_lab` (identical maths, but on a cache-sized block)."""
    if rgb01.dtype == np.uint8:
        lin = _SRGB8_TO_LINEAR[rgb01]
    else:
        rgb01 = rgb01.astype(np.float32)
        lin = np.where(rgb01 <= 0.04045, rgb01 / 12.92, ((rgb01 + 0.055) / 1.055) ** 2.4)
    xyz = (lin @ _M_RGB2XYZ.T) / _WHITE
    f = np.where(xyz > _DELTA**3, np.cbrt(xyz), xyz / (3 * _DELTA**2) + 4.0 / 29.0)
    return np.stack(
        [116.0 * f[..., 1] - 16.0, 500.0 * (f[..., 0] - f[..., 1]), 200.0 * (f[..., 1] - f[..., 2])],
        axis=-1,
    ).astype(np.float32)


def srgb_to_lab(rgb01: np.ndarray) -> np.ndarray:
    """rgb01: (..., 3) float in 0..1 -> Lab float32 (L 0..100, a/b roughly -128..127).

    A uint8 image (0..255) is accepted too and takes the exact lookup table. Images are converted
    band by band: the intermediate linear/XYZ arrays are the size of one band instead of the whole
    picture, which is what keeps a 4096x4096 conversion out of the allocator.
    """
    rgb = np.asarray(rgb01)
    if rgb.ndim != 3:
        return _srgb_block_to_lab(rgb)
    h = rgb.shape[0]
    out = np.empty((h, rgb.shape[1], 3), dtype=np.float32)

    def band(y0: int, y1: int) -> None:
        out[y0:y1] = _srgb_block_to_lab(rgb[y0:y1])

    run_bands(band, row_bands(h, rgb.shape[1]))
    return out


def _lab_block_to_srgb(lab: np.ndarray) -> np.ndarray:
    """One row band of `lab_to_srgb` (identical maths, but on a cache-sized block)."""
    h, w, _ = lab.shape
    f = np.empty_like(lab)
    fy = np.empty((h, w), dtype=np.float32)
    np.add(lab[..., 0], 16.0, out=fy)
    np.divide(fy, 116.0, out=fy)
    np.divide(lab[..., 1], 500.0, out=f[..., 0])
    np.add(f[..., 0], fy, out=f[..., 0])
    np.copyto(f[..., 1], fy)
    np.divide(lab[..., 2], 200.0, out=f[..., 2])
    np.subtract(fy, f[..., 2], out=f[..., 2])
    big = np.power(f, 3.0)
    small = np.empty_like(f)
    np.subtract(f, 4.0 / 29.0, out=small)
    np.multiply(small, 3 * _DELTA**2, out=small)
    np.copyto(small, big, where=(f > _DELTA))
    lin = small
    np.multiply(lin, _WHITE, out=lin)
    xyz = np.empty_like(lin)
    np.matmul(lin, _M_XYZ2RGB.T, out=xyz)
    lin = xyz
    np.clip(lin, 0.0, 1.0, out=lin)
    # encode back to sRGB, in place: both branches of the curve
    linear_sel = lin <= np.float32(0.0031308)
    encoded = np.empty_like(lin)
    np.power(lin, 1.0 / 2.4, out=encoded)
    np.multiply(encoded, 1.055, out=encoded)
    np.subtract(encoded, 0.055, out=encoded)
    np.multiply(lin, 12.92, out=lin)
    np.copyto(encoded, lin, where=linear_sel)
    return encoded


def lab_to_srgb(lab: np.ndarray) -> np.ndarray:
    """Lab -> (..., 3) float in 0..1, clipped to the sRGB gamut."""
    lab = np.asarray(lab, dtype=np.float32)
    if lab.ndim != 3:
        return _lab_block_to_srgb(lab)
    h = lab.shape[0]
    out = np.empty_like(lab)

    def band(y0: int, y1: int) -> None:
        out[y0:y1] = _lab_block_to_srgb(lab[y0:y1])

    run_bands(band, row_bands(h, lab.shape[1]))
    return out


def lab_delta(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Euclidean Lab distance (dE76) between (..., 3) arrays."""
    d = np.asarray(a, np.float32) - np.asarray(b, np.float32)
    return np.sqrt((d * d).sum(-1))


def luma(rgb: np.ndarray) -> np.ndarray:
    """Rec.709 luma from (..., 3) rgb in any common scale."""
    rgb = np.asarray(rgb)
    if rgb.dtype == np.uint8:
        # a byte is exact in float32, so the per-channel form is the generic one without the
        # intermediate (..., 3) float copy
        out = np.multiply(rgb[..., 0], np.float32(0.2126))
        tmp = np.multiply(rgb[..., 1], np.float32(0.7152))
        np.add(out, tmp, out=out)
        np.multiply(rgb[..., 2], np.float32(0.0722), out=tmp)
        np.add(out, tmp, out=out)
        return out
    rgb = np.asarray(rgb, dtype=np.float32)
    return 0.2126 * rgb[..., 0] + 0.7152 * rgb[..., 1] + 0.0722 * rgb[..., 2]
