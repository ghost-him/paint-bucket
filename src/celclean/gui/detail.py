"""The image maths behind the "噪点放大" (detail) view -- numpy only, no Qt.

Kept apart from :mod:`celclean.gui.view` so it can be imported and tested without PySide6::

    field = rgb - box_mean(rgb, DETAIL_RADIUS)      # per-pixel deviation, in colour levels
    shown = clip(128 + gain * field, 0, 255)

Why levels and not a contrast stretch:

* A global stretch (what the before/after figures do per crop) cannot work on a whole image --
  flat blocks exist at every brightness, so a pooled inter-quartile range spans the dark hat and
  the white background and the gain collapses to 1x.
* A local-std normalised version looks reasonable but destroys the comparison: the cleaned pane's
  small residual gets divided by its own (now tiny) spread and is amplified straight back up
  (measured: only 1.2x between the panes, i.e. exactly the difference hidden).
* The unnormalised residual shows the real amplitude, so the two panes compare directly: on the
  reference image the flat-block texture measures 0.489 levels before and 0.168 after (3x).
"""

from __future__ import annotations

import numpy as np

from ..ops import box_mean

MIN_ALPHA = 24
DETAIL_RADIUS = 2  # 5x5 neighbourhood; a wider one lets more of a gradient through as structure
DETAIL_MAX = 48.0  # field clip: past this everything is saturated anyway
DETAIL_TARGET = 12.0  # grey levels the flat-block texture should reach under the auto gain
GAIN_MIN, GAIN_MAX = 1, 64
FLAT_BLOCK = 32  # the block size "flat block" means everywhere else in this project


def detail_field(rgba: np.ndarray, radius: int = DETAIL_RADIUS) -> np.ndarray:
    """Local deviation field (H, W, 3) float16, in colour levels."""
    rgb = np.asarray(rgba)[..., :3].astype(np.float32)
    field = rgb - box_mean(rgb, radius)
    return np.clip(field, -DETAIL_MAX, DETAIL_MAX).astype(np.float16)


def render_detail(field: np.ndarray, gain: float, alpha: np.ndarray) -> np.ndarray:
    """Grey-centred rendering of a detail field, alpha carried over untouched."""
    out = np.empty((*field.shape[:2], 4), dtype=np.uint8)
    out[..., :3] = np.clip(128.0 + field.astype(np.float32) * gain, 0.0, 255.0).astype(np.uint8)
    out[..., 3] = alpha
    return out


def auto_gain(field: np.ndarray, valid: np.ndarray, target: float = DETAIL_TARGET) -> float:
    """Gain that brings the flat-block texture of `field` to about `target` grey levels.

    The flattest tenth of the 32x32 blocks is what "flat block" means everywhere else here, so the
    same selection is used instead of a global statistic that outlines dominate.
    """
    magnitude = np.abs(field.astype(np.float32)).mean(axis=2)
    ny, nx = magnitude.shape[0] // FLAT_BLOCK, magnitude.shape[1] // FLAT_BLOCK
    fallback = float(np.clip(target / max(float(magnitude.mean()), 0.02), GAIN_MIN, GAIN_MAX))
    if ny < 1 or nx < 1:
        return fallback
    tiles = magnitude[: ny * FLAT_BLOCK, : nx * FLAT_BLOCK]
    opaque = valid[: ny * FLAT_BLOCK, : nx * FLAT_BLOCK]
    shape = (ny, FLAT_BLOCK, nx, FLAT_BLOCK)
    level = tiles.reshape(shape).transpose(0, 2, 1, 3).mean(axis=(2, 3))
    full = opaque.reshape(shape).transpose(0, 2, 1, 3).all(axis=(2, 3))
    if not full.any():
        return fallback
    cutoff = float(np.percentile(level[full], 10))
    flat = full & (level <= cutoff)
    base = float(level[flat].mean()) if flat.any() else float(level[full].mean())
    return float(np.clip(target / max(base, 0.02), GAIN_MIN, GAIN_MAX))
