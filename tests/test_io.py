"""The image file layer: what the tool writes back out."""

import numpy as np
from PIL import Image

from celclean.io import load_rgba, save_rgba


def test_an_opaque_result_is_written_without_a_redundant_alpha_channel(tmp_path) -> None:
    opaque = np.zeros((8, 8, 4), np.uint8)
    opaque[..., :3] = (200, 60, 90)
    opaque[..., 3] = 255
    target = save_rgba(opaque, tmp_path / "opaque.png")
    assert Image.open(target).mode == "RGB"
    assert np.array_equal(load_rgba(target)[..., :3], opaque[..., :3])


def test_a_transparent_result_keeps_its_alpha_channel(tmp_path) -> None:
    transparent = np.zeros((8, 8, 4), np.uint8)
    transparent[..., 3] = 128
    target = save_rgba(transparent, tmp_path / "alpha.png")
    assert Image.open(target).mode == "RGBA"
    assert np.array_equal(load_rgba(target), transparent)


def test_load_rgba_normalises_other_modes(tmp_path) -> None:
    grey = Image.new("L", (4, 4), 128)
    path = tmp_path / "grey.png"
    grey.save(path)
    loaded = load_rgba(path)
    assert loaded.shape == (4, 4, 4) and (loaded[..., 3] == 255).all()
