"""Behavioural tests: each one defends a property the tool is supposed to have."""

from __future__ import annotations

import numpy as np
import pytest

from PIL import Image

from celclean import Options, clean, flatten, parse_color
from celclean.qa import image_metrics
from celclean.ops import any_in_window, box_count, box_mean, masked_box_mean


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


def test_masked_box_mean_2d_input_and_mask_semantics():
    """A 2-D map (e.g. luma) must be accepted, and the mask must keep invalid pixels out.

    `x` was multiplied against `valid[..., None]`, so a 2-D `x` broadcast into an (h, w, h) array:
    a 1254x1254 call asked for 7.35 GiB. This test also pins the mask contract on a 64x64 map.
    """
    x = np.zeros((64, 64), np.float32)
    x[:8, :8] = 1e6  # garbage outside the mask
    valid = np.ones((64, 64), bool)
    valid[:8, :8] = False

    out = masked_box_mean(x, valid, 4)
    assert out.shape == (64, 64, 1)
    assert np.all(out == 0.0)  # the garbage never leaks, and nothing divides by a zero count

    full = masked_box_mean(x, np.ones((64, 64), bool), 4)
    assert np.array_equal(full[..., 0], box_mean(x, 4))  # an all-valid mask is the plain box mean


def test_any_in_window_matches_the_counted_equivalent():
    """The alpha stage asks "is there a non-opaque pixel in this 3x3 window"; this pins that OR == count != 0."""
    mask = np.random.default_rng(5).random((40, 40)) < 0.2
    for r in (1, 2):
        assert np.array_equal(any_in_window(mask, r), box_count(mask, r) != 0)


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


def test_alpha_below_min_keeps_its_colour():
    """Pixels under MIN_ALPHA never enter any average, so the Lab round trip is meaningless for
    them — writing the Lab origin there blackened the semi-transparent fringe of the silhouette.
    Measured on the reference image before the fix: 1166 px changed by >8 levels, worst 21.4,
    none of which the masked metrics could see (they peg the mask at alpha >= 24)."""
    h = w = 64
    src = np.zeros((h, w, 4), dtype=np.uint8)
    src[..., 3] = 255
    src[..., :3] = (200, 120, 60)
    src[0:8, :, 3] = 20  # semi-transparent band: invisible on white when its colour survives
    src[0:8, :, :3] = (240, 240, 240)
    src[56:, 56:, 3] = 0  # fully transparent corner: colour must be cleared
    src[56:, 56:, :3] = (7, 7, 7)

    out, _ = clean(src, Options())

    assert tuple(out[2, 2, :3]) == (240, 240, 240)
    assert out[2, 2, 3] == 20
    assert tuple(out[60, 60, :3]) == (0, 0, 0)
    assert out[60, 60, 3] == 0


# ------------------------------------------------------------------ flat-block plane model


def _ramp_block(size=256, lo=100.0, hi=140.0, box=(16, 16, 224, 224)) -> np.ndarray:
    """A block whose green channel ramps linearly -- a "cheek shading" block."""
    x, y, w, h = box
    img = np.zeros((size, size, 4), dtype=np.uint8)
    img[y : y + h, x : x + w, 0] = 180
    img[y : y + h, x : x + w, 1] = np.rint(np.linspace(lo, hi, w)).astype(np.uint8)
    img[y : y + h, x : x + w, 2] = 150
    img[y : y + h, x : x + w, 3] = 255
    return img


def test_plane_snap_follows_a_ramp_that_constant_snap_flattens():
    """A ramp block is the one case where the constant repaint is catastrophic: it paints the whole
    ramp one colour (measured on a synthetic 40-level ramp: residual 11.1 vs 2.1 levels). The plane
    model must follow it instead."""
    img = _ramp_block()
    x, y, w, h = 16, 16, 224, 224
    row = slice(y + h // 2, y + h // 2 + 1)
    core = (row, slice(x + 4, x + w - 4))
    ideal = np.linspace(100.0, 140.0, w)[4:-4]

    flat, _ = clean(img, Options(radius=12, snap=True, snap_mode="constant", snap_tol=1.0))
    plane, _ = clean(img, Options(radius=12, snap=True, snap_mode="plane", snap_tol=1.0))

    flat_row = flat[core][0, :, 1].astype(np.float64)
    plane_row = plane[core][0, :, 1].astype(np.float64)
    assert flat_row.std() < 2.0, "the constant model is expected to flatten the ramp"
    assert plane_row.std() > 0.8 * ideal.std(), "the plane model must keep the ramp"
    assert np.corrcoef(plane_row, ideal)[0, 1] > 0.98


def test_component_planes_recovers_an_exact_plane():
    from celclean.ops import component_planes

    lab = np.zeros((24, 32, 3), np.float32)
    yy, xx = np.mgrid[0:24, 0:32]
    lab[..., 0] = 10.0 + 0.5 * yy + 0.25 * xx
    lab[..., 1] = 20.0 - 0.3 * yy
    lab[..., 2] = 5.0 + 0.1 * xx
    labels = np.full((24, 32), -1, np.int32)
    labels[4:20, 4:28] = 0

    offs, slps, ctr, rms = component_planes(lab, labels, 1)

    assert float(rms[0]) < 1e-3
    assert abs(float(offs[0, 0]) - (10.0 + 0.5 * ctr[0, 0] + 0.25 * ctr[0, 1])) < 1e-3
    assert abs(float(slps[0, 0, 0]) - 0.5) < 1e-3  # d/dy
    assert abs(float(slps[0, 1, 0]) - 0.25) < 1e-3  # d/dx
    assert abs(float(slps[0, 0, 1]) + 0.3) < 1e-3
    assert abs(float(slps[0, 1, 2]) - 0.1) < 1e-3


def test_plane_snap_leaves_a_non_planar_block_alone():
    """A block that a plane cannot describe (here: strong curvature) must not be repainted -- the
    relative-residual guard rejects it while the surrounding flat blocks are still repainted."""
    from celclean.ops import component_planes

    size = 128
    yy, xx = np.mgrid[0:size, 0:size]
    curved = (60.0 + 40.0 * ((yy / size - 0.5) ** 2 * 4)).astype(np.float32)
    lab = np.stack([curved, curved, curved], axis=-1)
    labels = np.full((size, size), -1, np.int32)
    labels[8:120, 8:120] = 0
    offs, slps, ctr, rms = component_planes(lab, labels, 1)
    assert float(rms[0]) > 2.0, "a curved block is not planar"


def _silhouette() -> np.ndarray:
    """Opaque square in the middle, a 1 px anti-aliasing ring, transparent elsewhere.

    The transparent pixels carry the garbage colour real cut-outs have (black here, white there),
    and the ring is the blend that must survive.
    """
    img = np.zeros((32, 32, 4), np.uint8)
    img[8:24, 8:24] = (200, 60, 90, 255)
    img[7, 7:25] = img[24, 7:25] = (200, 60, 90, 128)
    img[7:25, 7] = img[7:25, 24] = (200, 60, 90, 128)
    img[..., 3][0, 0] = 0
    img[0, 0, :3] = (255, 255, 255)  # garbage white
    img[31, 31, :3] = (0, 0, 0)
    return img


def test_flatten_composites_onto_the_colour_instead_of_painting_it():
    """Transparent pixels must land exactly on the colour, whatever garbage their RGB holds."""
    colour = (12, 34, 56)
    out = flatten(_silhouette(), colour)
    assert (out[..., 3] == 255).all()  # opaque
    assert tuple(out[0, 0, :3]) == colour  # was garbage white
    assert tuple(out[31, 31, :3]) == colour  # was garbage black
    assert tuple(out[1, 1, :3]) == colour


def test_flatten_keeps_the_anti_aliasing_ramp_light():
    """A half-transparent black pixel over white is ~127, not 0: no black fringe."""
    img = np.zeros((4, 4, 4), np.uint8)
    img[..., :3] = (0, 0, 0)
    img[..., 3] = 0
    img[1, 1] = (0, 0, 0, 128)
    out = flatten(img, (255, 255, 255))
    assert abs(int(out[1, 1, 0]) - 127) <= 1
    assert tuple(out[1, 1, :3]) == (127, 127, 127) or tuple(out[1, 1, :3]) == (128, 128, 128)


def test_flatten_agrees_with_pillows_alpha_composite():
    """Two independent compositors, one contract (PIL rounds in integer math, hence the +-1)."""
    rng = np.random.default_rng(3)
    img = rng.integers(0, 256, (48, 48, 4), dtype=np.uint8)
    for colour in ((255, 255, 255), (0, 0, 0), (12, 34, 56)):
        ours = flatten(img, colour)[..., :3].astype(np.int16)
        bg = Image.new("RGBA", (48, 48), (*colour, 255))
        theirs = np.asarray(Image.alpha_composite(bg, Image.fromarray(img, "RGBA")))[..., :3].astype(np.int16)
        assert np.abs(ours - theirs).max() <= 1


def test_flatten_via_the_pipeline_only_adds_the_background():
    """End to end: flattening an image leaves every opaque pixel alone."""
    rgb = mottled_flat_block(np.array([180, 120, 90]), size=96, seed=5)
    rgba = rgba_from(rgb, alpha=255)
    rgba[:20, :20, 3] = 0  # a transparent corner
    rgba[:20, :20, :3] = 0
    clear, _ = clean(rgba, Options(alpha_mode="normalize"))
    flat, info = clean(rgba, Options(alpha_mode="flatten", bg_color=(20, 40, 60)))
    assert info["alpha_mode"] == "flatten" and info["bg_color"] == [20, 40, 60]
    assert (flat[..., 3] == 255).all()
    assert tuple(flat[5, 5, :3]) == (20, 40, 60)
    opaque = rgba[..., 3] == 255
    assert np.array_equal(flat[..., :3][opaque], clear[..., :3][opaque])


def test_parse_color_accepts_the_forms_the_cli_documents():
    assert parse_color("#101820") == (16, 24, 32)
    assert parse_color("101820") == (16, 24, 32)
    assert parse_color("#f00") == (255, 0, 0)
    assert parse_color("white") == (255, 255, 255)
    assert parse_color(" BLACK ") == (0, 0, 0)
    assert parse_color((1, 2, 3)) == (1, 2, 3)
    for bad in ("", "#12345", "rgb(1,2,3)", "teal", (1, 2, 300)):
        with pytest.raises(ValueError):
            parse_color(bad)


def test_the_report_composites_the_original_for_a_flattened_output():
    """Otherwise the intended background replacement would dominate every change statistic."""
    rgb = mottled_flat_block(np.array([200.0, 200.0, 200.0]), size=64, seed=1)
    rgba = rgba_from(rgb, alpha=255)
    rgba[20:44, 20:44, 3] = 90  # a half-transparent patch with a dark RGB, as real cut-outs have
    rgba[20:44, 20:44, :3] = 40
    flat, _ = clean(rgba, Options(alpha_mode="flatten", bg_color=(255, 255, 255)))

    naive = image_metrics(rgba, flat)
    honest = image_metrics(rgba, flat, composite_original=(255, 255, 255))
    assert naive["max_abs_delta_levels"] > 50  # the patch lighting up is the point of the mode
    assert honest["max_abs_delta_levels"] < naive["max_abs_delta_levels"] / 5
    assert honest["original_composited"] == [255, 255, 255] and naive["original_composited"] is None
    # the content mask is the raw original's in both cases, so the numbers stay comparable
    assert honest["pixels_valid"] == naive["pixels_valid"]
