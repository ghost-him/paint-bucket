"""The cleanup pipeline: RGBA in -> RGBA out.

Stages (see README for the plain-language version):

  S0  mask        pixels with alpha < 24 are not real image content (their RGB is garbage)
  S1  noise scale robust per-pixel sigma, used for every threshold below
  S2  core        range-weighted local mean in Lab ("surface blur") - removes grain and the
                  low-frequency mottle while keeping edges, gradients, thin strokes and AA
  S3  repaint     every connected flat block is replaced by its own mean colour, which is what
                  makes a block exactly one colour (a local average cannot do that)
  S4  alpha       interior alpha -> 255, stray near-zero alpha -> 0, RGB of transparent -> 0
"""

from __future__ import annotations

from dataclasses import dataclass, asdict

import numpy as np

from .color import lab_to_srgb, luma, srgb_to_lab
from .ops import (
    any_in_window,
    bilateral,
    component_means,
    dilate,
    erode,
    grow_labels,
    label_components,
    masked_box_mean,
    merge_close_labels,
    robust_sigma_luma,
)

#: nominal pixel count the default radius is tuned for (1254x1254 = the reference image)
REF_PIXELS = 1254 * 1254
REF_RADIUS = 16
REF_SIGMA_RANGE = 1.5
MIN_ALPHA = 24  # below this the RGB channel is meaningless


@dataclass
class Options:
    strength: float = 1.0  # multiplies the range sigma (1.0 = default, higher = flatter)
    snap: bool = False  # repaint each connected flat block with its own mean colour
    snap_tol: float = 0.5  # max distance (dE) from the local mean that still counts as flat
    min_block: int = 64  # blocks smaller than this many px are left to the denoiser
    merge_tol: float = 1.0  # adjacent blocks closer than this (dE) share one colour
    close_radius: int = 2  # morphological closing radius for the flat mask (fills pinholes)
    radius: int | None = None  # None = auto from image size
    stride: int = 2  # offset subsampling inside the bilateral window
    sigma_grain: float | None = None  # None = measure it
    alpha_mode: str = "normalize"  # "normalize" | "keep"
    aa_band: int = 2  # px of the flat mask eroded before snapping (protects anti-aliasing)


def auto_radius(width: int, height: int) -> int:
    scale = (width * height / REF_PIXELS) ** 0.5
    return int(min(24, max(3, round(REF_RADIUS * scale))))


def clean(rgba: np.ndarray, opts: Options | None = None) -> tuple[np.ndarray, dict]:
    """Clean an RGBA uint8 image. Returns (out_rgba_uint8, info)."""
    opts = opts or Options()
    src = np.asarray(rgba)
    if src.ndim != 3 or src.shape[2] != 4:
        raise ValueError("expected an HxWx4 RGBA image")
    h, w, _ = src.shape

    alpha = src[..., 3].astype(np.float32)
    alpha_raw = src[..., 3].copy()
    valid = alpha >= MIN_ALPHA

    lab = srgb_to_lab(src[..., :3])  # uint8 in -> exact sRGB->linear table inside
    np.multiply(lab, valid[..., None], out=lab)  # transparent pixels must not leak into the means

    # ---- S1 noise scale
    sigma_lab = opts.sigma_grain
    if sigma_lab is None:
        sigma_lab = robust_sigma_luma(lab[..., 0], valid)
    sigma_srgb = robust_sigma_luma(luma(src[..., :3]), valid)

    radius = opts.radius if opts.radius is not None else auto_radius(w, h)
    sigma_range = float(np.clip(REF_SIGMA_RANGE * opts.strength, 0.5, 8.0))

    # ---- S2 core
    out_lab = bilateral(lab, valid, radius, sigma_range, stride=opts.stride)

    # ---- S3 repaint flat blocks with exactly one colour
    # A local average cannot remove variation larger than its window, so the low-frequency
    # "cloud" survives it. Here every connected flat block is instead replaced by its own mean
    # colour, which is what actually makes a block one colour.
    flat = np.zeros((h, w), dtype=bool)
    snapped_px = 0
    snapped_blocks = 0
    if opts.snap:
        r_dev = max(4, radius * 2 // 3)
        local_mean = masked_box_mean(out_lab, valid, r_dev)
        flat = valid & (np.abs(out_lab - local_mean).max(-1) <= opts.snap_tol)
        if opts.close_radius:
            flat = erode(dilate(flat, opts.close_radius), opts.close_radius)  # fill pinholes
        flat = erode(flat, opts.aa_band)  # keep the anti-aliased border of every block intact
        labels, count = label_components(flat)
        if count:
            sizes = np.bincount(labels.ravel()[labels.ravel() >= 0], minlength=count)
            keep = sizes >= opts.min_block
            means = component_means(out_lab, labels, count)
            if keep.any():
                # grow each block over the rest of the image, but only over pixels that are still
                # within snap_tol of the block's mean: the block reaches its true border and every
                # pixel in it finally gets exactly the same colour
                labels = grow_labels(out_lab, labels, means, valid & keep[labels.clip(0)], opts.snap_tol)
                labels, means = merge_close_labels(labels, means, out_lab, opts.merge_tol)
                sel = labels >= 0
                out_lab = np.where(sel[..., None], means[labels.clip(0)], out_lab)
                snapped_px = int(sel.sum())
                snapped_blocks = int(means.shape[0])

    # ---- back to sRGB 8 bit
    out = np.empty_like(src)
    rgb01 = lab_to_srgb(out_lab)
    np.multiply(rgb01, 255.0, out=rgb01)
    np.rint(rgb01, out=rgb01)
    np.clip(rgb01, 0, 255, out=rgb01)
    out[..., :3] = rgb01.astype(np.uint8)
    out[..., 3] = alpha_raw
    # Below MIN_ALPHA the Lab round trip is meaningless (those pixels never enter any average),
    # so they currently come back as the Lab origin — i.e. black. Keeping the source colour is
    # what the alpha channel promises: an alpha=20 pixel over a white page is invisible, but a
    # blackened one is a visible grey dot (measured on this image: 1166 px with >8 levels of
    # change, worst 21.4, all of them this case — and all of them invisible to the masked
    # metrics below, which is why they are also reported as "rendered" numbers now).
    np.copyto(out[..., :3], src[..., :3], where=(~valid)[..., None])

    # ---- S4 alpha channel
    interior = (alpha >= 248) & ~any_in_window(alpha < 248, 1)
    stray = (alpha <= 2) & ~any_in_window(alpha > 2, 1)
    alpha_out = alpha_raw.copy()
    if opts.alpha_mode == "normalize":
        alpha_out[interior] = 255
    alpha_out[stray] = 0
    out[..., 3] = alpha_out
    np.copyto(out[..., :3], np.uint8(0), where=(alpha_out == 0)[..., None])

    info = {
        "size": [w, h],
        "sigma_grain_lab": round(float(sigma_lab), 4),
        "sigma_grain_srgb_levels": round(float(sigma_srgb), 4),
        "radius": radius,
        "sigma_range": round(sigma_range, 3),
        "stride": opts.stride,
        "snap": opts.snap,
        "snapped_px": snapped_px,
        "snapped_blocks": snapped_blocks,
        "alpha_mode": opts.alpha_mode,
        "alpha_interior_snapped_px": int((interior & (alpha_raw != 255)).sum()),
        "alpha_stray_zeroed_px": int((stray & (alpha_raw != 0)).sum()),
        "options": asdict(opts),
    }
    return out, info
