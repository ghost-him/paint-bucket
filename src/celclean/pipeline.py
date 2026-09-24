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
    component_planes,
    dilate,
    erode,
    grow_labels,
    grow_planes,
    label_components,
    masked_box_mean,
    merge_close_labels,
    merge_close_planes,
    plane_field,
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
    # 1 = every tap of the window (default). Measured on the reference image at equal wall time,
    # stride 1 with a smaller radius beats stride 2 with a bigger one on every purity axis
    # (flat-pixel variation 0.0591 vs 0.0758, uniform 3x3 neighbourhoods 0.528 vs 0.496 at ~2 s),
    # and it never touched the real gradients (blush correlation 0.9993 for every setting).
    stride: int = 1  # offset subsampling inside the bilateral window
    sigma_grain: float | None = None  # None = measure it
    alpha_mode: str = "normalize"  # "normalize" | "keep" | "flatten"
    # only read by alpha_mode == "flatten": the solid colour transparency is composited onto
    bg_color: tuple[int, int, int] = (255, 255, 255)
    aa_band: int = 2  # px of the flat mask eroded before snapping (protects anti-aliasing)
    snap_mode: str = "plane"  # "plane" = fit a plane per block, "constant" = one colour per block
    plane_tol: float = 4.0  # keep a block if its plane residual <= plane_tol x the median residual
    slope_tol: float = 0.05  # max slope difference (dE/px) for two adjacent blocks to merge


NAMED_COLORS: dict[str, tuple[int, int, int]] = {"white": (255, 255, 255), "black": (0, 0, 0)}


def parse_color(value: str | tuple[int, int, int]) -> tuple[int, int, int]:
    """`#rrggbb`, `rrggbb`, `#rgb`, `white`/`black` or an RGB triple -> (r, g, b)."""
    if isinstance(value, (tuple, list)):
        channels = tuple(int(v) for v in value)
        if len(channels) != 3 or not all(0 <= v <= 255 for v in channels):
            raise ValueError(f"expected three channels in 0..255, got {value!r}")
        return channels  # type: ignore[return-value]
    text = str(value).strip().lower()
    if text in NAMED_COLORS:
        return NAMED_COLORS[text]
    digits = text[1:] if text.startswith("#") else text
    if len(digits) == 3:
        digits = "".join(c * 2 for c in digits)
    if len(digits) != 6 or any(c not in "0123456789abcdef" for c in digits):
        raise ValueError(f"expected #rrggbb (or white/black), got {value!r}")
    return (int(digits[0:2], 16), int(digits[2:4], 16), int(digits[4:6], 16))


def flatten(rgba: np.ndarray, color: tuple[int, int, int] = (255, 255, 255)) -> np.ndarray:
    """Composite the image onto a solid colour and make it fully opaque.

    A real composite, not "paint the transparent pixels": the RGB under transparency is garbage
    (after the alpha stage it is black), and the 1-2 px anti-aliasing ramp is exactly the blend
    that has to survive.  Painting instead of compositing leaves that ramp dark -- the classic
    black fringe.  `np.rint` keeps the result from drifting half a level darker.
    """
    src = np.asarray(rgba)
    alpha = (src[..., 3].astype(np.float32) / 255.0)[..., None]
    background = np.asarray(color, dtype=np.float32)[None, None, :]
    mixed = src[..., :3].astype(np.float32) * alpha
    mixed += (1.0 - alpha) * background
    out = np.empty_like(src)
    out[..., :3] = np.clip(np.rint(mixed), 0, 255).astype(np.uint8)
    out[..., 3] = 255
    return out


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
            if opts.snap_mode == "plane":
                # A block is not always one colour: repainting a ramp with a constant turns it into a
                # staircase (the measured cost of the constant mode). A plane follows the ramp and
                # still averages the zero-mean mottle away; blocks whose plane leaves a residual are
                # left to the denoiser, and blocks that leaked across a real edge are rejected by the
                # same test.
                offs, slps, ctr, rms = component_planes(out_lab, labels, count)
                # "Planar" is judged RELATIVE to the other blocks, not by an absolute number: the
                # residual of a typical block is the leftover cloud, i.e. exactly what we want to
                # average away, while a curved shading ramp or a block that leaked across an edge
                # fits far worse. An absolute threshold would do the opposite and reject the
                # noisiest - i.e. the most in-need-of-repainting - blocks (measured: with tol=1.0
                # the 8 flattest tiles went from 7/8 clean to 4/8).
                ref = float(np.median(rms[keep])) if keep.any() else 0.0
                tol = max(0.5, opts.plane_tol * ref)
                seeds = valid & keep[labels.clip(0)] & (rms <= tol)[labels.clip(0)]
                if seeds.any():
                    labels = grow_planes(out_lab, labels, offs, slps, ctr, seeds, opts.snap_tol)
                    labels, offs, slps, ctr = merge_close_planes(
                        labels, offs, slps, ctr, out_lab, opts.merge_tol, opts.slope_tol
                    )
                    sel = labels >= 0
                    planes = plane_field((h, w), labels.clip(0), offs, slps, ctr)
                    out_lab = np.where(sel[..., None], planes, out_lab)
                    snapped_px = int(sel.sum())
                    snapped_blocks = int(offs.shape[0])
            else:
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

    # ---- S5 optional: give the transparency a solid background (fully opaque output)
    if opts.alpha_mode == "flatten":
        out = flatten(out, opts.bg_color)

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
        "bg_color": list(opts.bg_color),
        "alpha_interior_snapped_px": int((interior & (alpha_raw != 255)).sum()),
        "alpha_stray_zeroed_px": int((stray & (alpha_raw != 0)).sum()),
        "options": asdict(opts),
    }
    return out, info
