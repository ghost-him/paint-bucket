"""The maths of the GUI's detail view (Qt-free, so no PySide6 needed to run these)."""

import numpy as np

from celclean.gui.detail import auto_gain, detail_field


def _flat_with_grain(noise: float, size: int = 128, seed: int = 0) -> np.ndarray:
    """A grey block whose only variation is zero-mean grain of `noise` levels."""
    rng = np.random.default_rng(seed)
    rgba = np.zeros((size, size, 4), np.uint8)
    rgba[..., :3] = np.clip(200.0 + rng.normal(0.0, noise, (size, size, 3)), 0, 255).astype(np.uint8)
    rgba[..., 3] = 255
    return rgba


def test_detail_field_is_flat_on_a_pure_ramp() -> None:
    # a symmetric window sits exactly on a linear ramp, so a gradient must not light up:
    # otherwise the view would paint real shading as if it were noise
    ramp = np.zeros((64, 64, 4), np.uint8)
    ramp[..., :3] = np.arange(64)[None, :, None]
    ramp[..., 3] = 255
    field = detail_field(ramp).astype(np.float32)
    assert np.abs(field[8:-8, 8:-8]).max() < 1e-3


def test_detail_field_keeps_the_grain_amplitude() -> None:
    # the point of the view: what it shows has the size of the real residual
    for noise in (0.5, 2.0):
        field = np.abs(detail_field(_flat_with_grain(noise)).astype(np.float32))
        measured = float(field[8:-8, 8:-8].mean())
        assert 0.6 * noise < measured < 1.2 * noise


def test_auto_gain_scales_the_flat_block_texture_up() -> None:
    # Floor to keep in mind: 8-bit cannot hold sub-level grain, so even a perfectly clean flat
    # block carries ~0.3 levels of rounding texture and lands around gain 25 (see TECHNICAL 9).
    quiet = _flat_with_grain(0.5)
    mid = _flat_with_grain(1.0)
    loud = _flat_with_grain(3.0)
    gains = [auto_gain(detail_field(im), im[..., 3] >= 24) for im in (quiet, mid, loud)]
    assert gains[0] > gains[1] > gains[2]
    assert 10 <= gains[1] <= 20  # ~12 levels / ~0.8 measured texture
    assert 3 <= gains[2] <= 8


def test_auto_gain_survives_an_image_smaller_than_a_block() -> None:
    tiny = _flat_with_grain(1.0, size=16)
    gain = auto_gain(detail_field(tiny), tiny[..., 3] >= 24)
    assert 1 <= gain <= 64
