"""Measurement and before/after figures.

`image_metrics` implements the acceptance protocol from the design document, so the
numbers reported by `celclean clean --report` can be compared across runs.
`write_comparison` produces the overview + detail figures used in the README.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw

from .color import lab_delta, luma, srgb_to_lab
from .pipeline import flatten
from .ops import box_mean, gradient_magnitude, local_spread, robust_sigma_luma

FLAT_TILE = 32
MIN_ALPHA = 24


def _rgba_to_np(img: Image.Image) -> np.ndarray:
    if img.mode != "RGBA":
        img = img.convert("RGBA")
    return np.asarray(img)


def _median_filter_radius9_scalar(x: np.ndarray, chunk: int = 192) -> np.ndarray:
    """Median of a 9x9 window for a (H, W) float32 field, computed in row bands."""
    h, w = x.shape
    pad = np.pad(x, 4, mode="edge")
    out = np.empty_like(x)
    for y0 in range(0, h, chunk):
        y1 = min(h, y0 + chunk)
        windows = np.stack(
            [pad[y0 + i : y1 + i, j : j + w] for i in range(9) for j in range(9)], axis=-1
        )
        out[y0:y1] = np.median(windows, axis=-1)
    return out


def image_metrics(
    orig_rgba: np.ndarray,
    out_rgba: np.ndarray,
    tiles: int = 8,
    background: tuple[int, int, int] = (255, 255, 255),
    composite_original: tuple[int, int, int] | None = None,
) -> dict:
    """The measured acceptance numbers (see README "如何验收").

    `background` is the page both images are composited on for the "rendered" rows (white by
    default; a flattened output should be measured on its own background).

    `composite_original`, when given, first composites the ORIGINAL onto that colour, so a flattened
    output is compared like for like.  Without it the half-transparent pixels would look like a
    199-level change (the background replacement this mode is *for*) and would hide the real
    question: how gently the opaque content moved.  The content mask stays the original one
    (alpha >= 24 on the raw original) in both cases, so the numbers remain comparable with the
    published baseline.
    """
    orig = np.asarray(orig_rgba)
    out = np.asarray(out_rgba)
    if orig.shape != out.shape:
        raise ValueError("images must have the same shape")
    h, w, _ = orig.shape
    valid = orig[..., 3] >= MIN_ALPHA
    if composite_original is not None:
        orig = flatten(orig, composite_original)

    lo = luma(orig[..., :3].astype(np.float32))
    lc = luma(out[..., :3].astype(np.float32))
    delta = np.abs(lc - lo)

    # flattest tiles of the ORIGINAL: where the "should be one colour" claim lives.
    # Select by tile std (a ramp has a large std, an edge a huge one, a mottled flat block a
    # small one), require the tile to be fully opaque, then report how the cleaning changed it.
    ny, nx = (h - 8) // FLAT_TILE, (w - 8) // FLAT_TILE
    tiles_flat = lo[4 : 4 + ny * FLAT_TILE, 4 : 4 + nx * FLAT_TILE].reshape(ny, FLAT_TILE, nx, FLAT_TILE)
    full = valid[4 : 4 + ny * FLAT_TILE, 4 : 4 + nx * FLAT_TILE].reshape(ny, FLAT_TILE, nx, FLAT_TILE)
    score = np.where(full.all(axis=(1, 3)), tiles_flat.std(axis=(1, 3)), np.inf)
    order = np.argsort(score.ravel())[:tiles]

    tile_rows = []
    for i in order:
        by, bx = divmod(int(i), nx)
        y, x = by * FLAT_TILE + 4, bx * FLAT_TILE + 4
        o = lo[y : y + FLAT_TILE, x : x + FLAT_TILE]
        c = lc[y : y + FLAT_TILE, x : x + FLAT_TILE]
        tile_rows.append(
            {
                "x": int(x),
                "y": int(y),
                "std_before": round(float(o.std()), 3),
                "std_after": round(float(c.std()), 3),
                "bias_levels": round(float(c.mean() - np.median(o)), 3),
            }
        )

    res = np.abs(lo[1:-1, 1:-1] - _median_filter_radius9_scalar(lo)[1:-1, 1:-1])
    flat_sel = (res < 0.8) & valid[1:-1, 1:-1]
    res_after = np.abs(lc[1:-1, 1:-1] - _median_filter_radius9_scalar(lc)[1:-1, 1:-1])
    lab_o = srgb_to_lab(orig[..., :3].astype(np.float32) / 255.0)
    lab_c = srgb_to_lab(out[..., :3].astype(np.float32) / 255.0)

    v = valid.copy()
    v[out[..., 3] == 0] = False
    # Coverage for everything the masked numbers cannot see: alpha compositing on a white page
    # over EVERY pixel. The masked rows above once hid a real artefact class (alpha in [8,24)
    # whose RGB was written black: 1166 px changed by >8 levels, worst 21.4 — invisible above).
    dren = np.abs(_rendered_luma(out, background) - _rendered_luma(orig, background))
    return {
        "size": [w, h],
        "pixels_valid": int(v.sum()),
        "render_background": [int(v) for v in background],
        "original_composited": None if composite_original is None else [int(v) for v in composite_original],
        "rendered_max_abs_delta_levels": round(float(dren.max()), 3),
        "rendered_mean_abs_delta_levels": round(float(dren.mean()), 3),
        "rendered_p99_abs_delta_levels": round(float(np.percentile(dren, 99)), 3),
        "rendered_frac_gt8_levels": round(float((dren > 8).mean()), 5),
        "sigma_grain_before_levels": round(float(robust_sigma_luma(lo, valid)), 4),
        "sigma_grain_after_levels": round(float(robust_sigma_luma(lc, v)), 4),
        "flat_tiles": tile_rows,
        "flat_tile_std_before_mean": round(float(np.mean([t["std_before"] for t in tile_rows])), 3),
        "flat_tile_std_after_mean": round(float(np.mean([t["std_after"] for t in tile_rows])), 3),
        "flat_tile_std_after_median": round(float(np.median([t["std_after"] for t in tile_rows])), 3),
        "flat_tiles_cleaned": int(sum(1 for t in tile_rows if t["std_after"] <= 0.1)),
        "flat_tile_bias_absmax": round(float(np.max([abs(t["bias_levels"]) for t in tile_rows])), 3),
        "local_variation_flat_px_before": round(float(res[flat_sel].mean()), 4),
        "local_variation_flat_px_after": round(float(res_after[flat_sel].mean()), 4),
        "corr_luma": round(float(np.corrcoef(lo[v], lc[v])[0, 1]), 6),
        "mean_abs_delta_levels": round(float(delta[v].mean()), 3),
        "p99_abs_delta_levels": round(float(np.percentile(delta[v], 99)), 3),
        "max_abs_delta_levels": round(float(delta[v].max()), 3),
        "frac_gt2_levels": round(float((delta[v] > 2).mean()), 5),
        "frac_gt8_levels": round(float((delta[v] > 8).mean()), 5),
        "max_deltaE_flat_px": round(
            float(np.percentile(lab_delta(lab_o, lab_c)[1:-1, 1:-1][flat_sel], 99)), 3
        ),
    }


def _page_label(m: dict) -> str:
    """How to name the background in :func:`metrics_text` (kept as "white page" for the default)."""
    bg = tuple(m.get("render_background", (255, 255, 255)))
    return "a white page" if bg == (255, 255, 255) else f"a {bg} page"


def metrics_text(m: dict) -> str:
    """Human-readable summary of :func:`image_metrics`, shared by the CLI and the GUI."""
    return "\n".join(
        [
            f"  flattest tiles ({len(m['flat_tiles'])}x 32x32): std {m['flat_tile_std_before_mean']} -> "
            f"{m['flat_tile_std_after_mean']} mean, {m['flat_tile_std_after_median']} median; "
            f"{m['flat_tiles_cleaned']} are now flat (std<=0.1); worst colour bias "
            f"{m['flat_tile_bias_absmax']} levels",
            f"  local variation on flat pixels: {m['local_variation_flat_px_before']} -> "
            f"{m['local_variation_flat_px_after']}",
            f"  grain sigma (levels): {m['sigma_grain_before_levels']} -> {m['sigma_grain_after_levels']}",
            f"  luma corr {m['corr_luma']}  mean|d| {m['mean_abs_delta_levels']}  "
            f"p99|d| {m['p99_abs_delta_levels']}  max|d| {m['max_abs_delta_levels']}",
            f"  pixels changed >2 levels {100 * m['frac_gt2_levels']:.2f}%  "
            f">8 levels {100 * m['frac_gt8_levels']:.2f}%",
            f"  on {_page_label(m)}, every pixel: max|d| {m['rendered_max_abs_delta_levels']}  "
            f"p99 {m['rendered_p99_abs_delta_levels']}  mean {m['rendered_mean_abs_delta_levels']}  "
            f">8 levels {100 * m['rendered_frac_gt8_levels']:.3f}%",
        ]
    )


def _solid_bg(np_rgba: np.ndarray, background: tuple[int, int, int] = (255, 255, 255)) -> Image.Image:
    img = Image.fromarray(np_rgba, "RGBA")
    bg = Image.new("RGBA", img.size, (int(background[0]), int(background[1]), int(background[2]), 255))
    return Image.alpha_composite(bg, img).convert("RGB")


def _rendered_luma(
    np_rgba: np.ndarray, background: tuple[int, int, int] = (255, 255, 255)
) -> np.ndarray:
    """Luma of the image as a viewer sees it on `background`, for every pixel including the
    transparent ones (alpha compositing, exact)."""
    a = np_rgba[..., 3].astype(np.float32) / 255.0
    rgb = np_rgba[..., :3].astype(np.float32)
    page = np.asarray(background, dtype=np.float32)[None, None, :]
    return luma(page - a[..., None] * (page - rgb))


def _boost(img: Image.Image) -> tuple[Image.Image, float]:
    """Auto-contrast around the median with a gain from the robust inter-quartile range.

    Makes sub-level mottle visible without letting the black/white parts of the crop decide
    the window (which is why a percentile stretch washes these crops out).
    """
    a = np.asarray(img).astype(np.float32)
    l = luma(a)
    p25, p50, p75 = np.percentile(l, [25, 50, 75])
    gain = float(np.clip(80.0 / max(p75 - p25, 1.0), 1.0, 24.0))
    out = np.clip((a - p50) * gain + 110.0, 0, 255).astype(np.uint8)
    return Image.fromarray(out), gain


def _label(img: Image.Image, text: str) -> Image.Image:
    out = img.copy()
    d = ImageDraw.Draw(out)
    d.rectangle([0, 0, 8 + 7 * len(text), 14], fill=(255, 255, 255))
    d.text((4, 3), text, fill=(0, 0, 0))
    return out


def pick_crops(orig_rgba: np.ndarray, out_rgba: np.ndarray, size: int = 256) -> list[tuple[int, int, int, int]]:
    """Two automatic crops: where the most mottle was removed, and a preserved gradient.

    Both are restricted to windows that are almost entirely opaque, so the figures never show
    a crop that is half transparent background.
    """
    orig = np.asarray(orig_rgba)
    out = np.asarray(out_rgba)
    lo = luma(orig[..., :3].astype(np.float32))
    lc = luma(out[..., :3].astype(np.float32))
    valid = orig[..., 3] >= MIN_ALPHA
    h, w = lo.shape
    r = size // 2

    lab_o = srgb_to_lab(orig[..., :3].astype(np.float32) / 255.0)
    lab_c = srgb_to_lab(out[..., :3].astype(np.float32) / 255.0)
    lab_o[~valid] = lab_c[~valid]  # keep transparent pixels out of the local statistics
    lab_c[~valid] = lab_o[~valid]

    spread_o = box_mean(local_spread(lab_o, 4), r)
    spread_c = box_mean(local_spread(lab_c, 4), r)
    grad = box_mean(gradient_magnitude(lc), r)
    damaged = box_mean(np.abs(lc - lo), r)
    cover = box_mean(valid.astype(np.float32), r)

    inside = cover > 0.995
    # most mottle removed, among windows that ARE flat in the original (excludes silhouettes/edges)
    cleaned = np.where(inside & (spread_o < 1.5), spread_o - spread_c, -1e9)
    # a real soft gradient that was left alone: visible slope, little pixel change, some shading
    kept = np.where(inside & (spread_o > 0.6) & (spread_o < 6.0), grad / (1.0 + damaged), -1e9)

    crops = []
    for field in (cleaned, kept):
        iy, ix = np.unravel_index(int(np.argmax(field)), field.shape)
        x = int(np.clip(ix - r, 0, w - size))
        y = int(np.clip(iy - r, 0, h - size))
        crops.append((x, y, size, size))
    return crops


def write_comparison(
    orig_rgba: np.ndarray,
    out_rgba: np.ndarray,
    out_dir: str | Path,
    crops: list[tuple[int, int, int, int]] | None = None,
    zoom: int = 3,
    overview_height: int = 700,
    background: tuple[int, int, int] = (255, 255, 255),
) -> list[Path]:
    """Write `overview.png` plus one figure per detail crop; returns the file paths."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    before = _solid_bg(np.asarray(orig_rgba), background)
    after = _solid_bg(np.asarray(out_rgba), background)
    written: list[Path] = []

    # ---- overview: whole image, before | after
    scale = min(1.0, overview_height / before.height)
    size = (max(1, int(before.width * scale)), max(1, int(before.height * scale)))
    b = _label(before.resize(size, Image.LANCZOS), "Before")
    a = _label(after.resize(size, Image.LANCZOS), "After")
    canvas = Image.new("RGB", (size[0] * 2 + 12, size[1]), (255, 255, 255))
    canvas.paste(b, (0, 0))
    canvas.paste(a, (size[0] + 12, 0))
    path = out_dir / "overview.png"
    canvas.save(path)
    written.append(path)

    # ---- detail crops: 1:1, zoomed, contrast-boosted
    if crops is None:
        crops = pick_crops(orig_rgba, out_rgba, size=min(256, min(orig_rgba.shape[:2]) // 2))
    for idx, (x, y, cw, ch) in enumerate(crops):
        x, y = max(0, int(x)), max(0, int(y))
        cw = min(int(cw), before.width - x)
        ch = min(int(ch), before.height - y)
        if cw <= 0 or ch <= 0:
            continue
        ob, oa = before.crop((x, y, x + cw, y + ch)), after.crop((x, y, x + cw, y + ch))
        bb, gain = _boost(ob)
        ba, _ = _boost(oa)
        rows = [
            ("1:1", ob, oa),
            (
                f"{zoom}x",
                ob.resize((cw * zoom, ch * zoom), Image.NEAREST),
                oa.resize((cw * zoom, ch * zoom), Image.NEAREST),
            ),
            (f"boost x{gain:.0f}", bb, ba),
        ]
        width = max(r[1].width for r in rows)
        height = sum(r[1].height for r in rows) + 16 * len(rows)
        canvas = Image.new("RGB", (width * 2 + 12, height), (255, 255, 255))
        yy = 0
        for label, left, right in rows:
            canvas.paste(_label(left, label), (0, yy))
            canvas.paste(_label(right, label), (width + 12, yy))
            yy += left.height + 16
        path = out_dir / f"detail-{idx + 1}-{x}_{y}.png"
        canvas.save(path)
        written.append(path)
    return written


def write_report(metrics: dict, info: dict, path: str | Path) -> Path:
    path = Path(path)
    path.write_text(
        json.dumps({"info": info, "metrics": metrics}, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    return path
