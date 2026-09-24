"""Behavioural tests: each one defends a property the tool is supposed to have."""

from __future__ import annotations

import numpy as np
import pytest

from celclean import Options, clean
from celclean.ops import box_mean


def rgba_from(rgb: np.ndarray, alpha: int | np.ndarray = 255) -> np.ndarray:
    h, w, _ = rgb.shape
    a = np.full((h, w), alpha, np.uint8) if np.isscalar(alpha) else np.asarray(alpha, np.uint8)
    out = np.empty((h, w, 4), np.uint8)
    out[..., :3] = np.clip(np.rint(rgb), 0, 255).astype(np.uint8)
    out[..., 3] = a
    return out


def luma(img: np.ndarray) -> np.ndarray:
    rgb = img[..., :3].astype(np.float32)
    return 0.2126 * rgb[..., 0] + 0.7152 * rgb[..., 1] + 0.0722 * rgb[..., 2]


def test_box_mean_is_unbiased():
    const = np.full((24, 24), 200.0, np.float32)
    assert np.allclose(box_mean(const, 5), 200.0)
    half = np.zeros((24, 24), np.float32)
    half[:, 12:] = 100.0
    # a window fully inside either half must return that half's value exactly
    assert np.allclose(box_mean(half, 5)[:, :7], 0.0)
    assert np.allclose(box_mean(half, 5)[:, 17:], 100.0)


def mottled_flat_block(color, size=160, grain=0.8, mottle=2.0, seed=0):
    """A flat colour block with the two defects the tool targets: grain + low-frequency mottle."""
    rng = np.random.default_rng(seed)
    field = box_mean(box_mean(rng.normal(0, 1, (size, size)), 6), 6)
    field = field / max(1e-6, field.std()) * mottle
    noisy = color[None, None, :] + field[..., None] + rng.normal(0, grain, (size, size, 3))
    return np.clip(noisy, 0, 255)


def test_grain_is_averaged_away():
    """The +-1 level grain every AI PNG carries: a local mean must crush it."""
    rng = np.random.default_rng(0)
    noisy = np.clip(np.array([70.0, 66.0, 80.0]) + rng.normal(0, 1.0, (160, 160, 3)), 0, 255)
    src = rgba_from(noisy)
    out, _ = clean(src, Options(radius=12, stride=1, snap=False))

    before = luma(src)[24:136, 24:136]
    after = luma(out)[24:136, 24:136]
    assert after.std() < 0.4 * before.std()
    assert np.percentile(after, 99) - np.percentile(after, 1) <= 1.5  # one value +- 8 bit rounding
    assert abs(after.mean() - before.mean()) < 0.4  # unbiased: no darkening/lightening


def test_mottle_is_reduced_without_bias():
    """The low-frequency "cloud" on flat blocks: reduced by the mean, and the block mean is kept."""
    src = rgba_from(mottled_flat_block(np.array([70.0, 66.0, 80.0]), mottle=2.0, grain=0.8))
    out, _ = clean(src, Options(radius=16, stride=2, snap=False))

    before = luma(src)[24:136, 24:136]
    after = luma(out)[24:136, 24:136]
    # a local mean reduces variation at or below its window scale; the rest needs --snap
    assert after.std() < 0.75 * before.std()
    assert abs(after.mean() - before.mean()) < 0.4


def test_soft_gradient_survives():
    rng = np.random.default_rng(1)
    ramp = np.linspace(200.0, 235.0, 200)[None, :, None] * np.ones((1, 1, 3))
    noisy = np.clip(ramp + rng.normal(0, 2.0, (80, 200, 3)), 0, 255)
    src = rgba_from(noisy)
    out, _ = clean(src, Options(radius=12, stride=1))

    row_before = luma(src)[40]
    row_after = luma(out)[40]
    assert np.corrcoef(row_before, row_after)[0, 1] > 0.99
    assert abs(np.ptp(row_after) - np.ptp(row_before)) < 0.1 * np.ptp(row_before)


def test_thin_stroke_keeps_its_contrast():
    rng = np.random.default_rng(2)
    base = np.full((140, 140, 3), 200.0)
    base[:, 68:70] = 120.0  # 2 px wide stroke
    noisy = np.clip(base + rng.normal(0, 2.0, base.shape), 0, 255)
    src = rgba_from(noisy)
    out, _ = clean(src, Options(radius=12, stride=1))

    before = luma(src)[70]
    after = luma(out)[70]
    contrast_before = np.median(before[:60]) - before.min()
    contrast_after = np.median(after[:60]) - after.min()
    assert contrast_after > 0.7 * contrast_before


def test_edge_stays_where_it_was():
    base = np.zeros((120, 120, 3))
    base[:, :] = 60.0
    base[:, 60:] = 200.0
    src = rgba_from(base)
    out, _ = clean(src, Options(radius=12, stride=1))

    row = luma(out)[60]
    assert abs(float(np.argmax(row > 130)) - 60) <= 1
    assert row[:58].max() - row[:58].min() < 2.0  # no bleed into the dark side
    assert row[62:].max() - row[62:].min() < 2.0


def test_alpha_interior_normalised_and_transparent_cleared():
    base = np.full((100, 100, 3), 200.0)
    alpha = np.zeros((100, 100), np.uint8)
    alpha[20:80, 20:80] = 253
    src = rgba_from(base, alpha)
    src[0, 0] = (255, 255, 255, 1)  # stray near-transparent pixel with garbage colour

    out, info = clean(src, Options(radius=8, stride=1))
    assert out[0, 0, 3] == 0
    assert out[0, 0, :3].tolist() == [0, 0, 0]
    assert out[50, 50, 3] == 255
    assert info["alpha_stray_zeroed_px"] == 1

    kept, _ = clean(src, Options(radius=8, stride=1, alpha_mode="keep"))
    assert kept[50, 50, 3] == 253


def test_transparent_pixels_do_not_leak_into_the_image():
    """The background RGB of AI PNGs is garbage (often pure white); it must not pull colour in."""
    rgb = np.full((100, 100, 3), 240.0)
    rgb[:, :30] = 255.0  # garbage background colour
    alpha = np.full((100, 100), 0, np.uint8)
    alpha[:, 30:] = 255
    src = rgba_from(rgb, alpha)

    out, _ = clean(src, Options(radius=12, stride=1))
    block = luma(out)[:, 45:]
    assert abs(block.mean() - 240.0) < 0.75
    assert block.max() - block.min() < 2.0


def test_snap_makes_a_flat_block_one_colour():
    src = rgba_from(mottled_flat_block(np.array([90.0, 92.0, 110.0]), size=120, mottle=1.2, seed=3))

    plain, _ = clean(src, Options(radius=10, stride=1, snap=False))
    snapped, info = clean(src, Options(radius=10, stride=1, snap=True, min_block=64))
    assert info["snapped_px"] > 0.5 * 120 * 120
    assert luma(snapped)[30:90, 30:90].std() < 0.02  # one exact colour for the whole block
    assert luma(plain)[30:90, 30:90].std() < 0.75 * luma(src)[30:90, 30:90].std()
    assert abs(luma(snapped)[30:90, 30:90].mean() - luma(src)[30:90, 30:90].mean()) < 0.6


@pytest.mark.parametrize("strength", [0.5, 1.0, 2.0])
def test_strength_stays_sane(strength: float):
    rng = np.random.default_rng(4)
    ramp = np.linspace(180.0, 220.0, 160)[None, :, None] * np.ones((1, 1, 3))
    noisy = np.clip(ramp + rng.normal(0, 2.0, (80, 160, 3)), 0, 255)
    src = rgba_from(noisy)
    out, info = clean(src, Options(radius=12, stride=1, strength=strength))
    assert info["sigma_range"] == pytest.approx(min(8.0, max(0.5, 1.5 * strength)))
    assert np.corrcoef(luma(src)[40], luma(out)[40])[0, 1] > 0.98
