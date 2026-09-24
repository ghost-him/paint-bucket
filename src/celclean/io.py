"""Reading and writing image files -- the only place that touches Pillow's file layer.

Output rule: a fully opaque image is written as **RGB**.  A redundant all-255 alpha channel adds
bytes and claims a transparency that is not there, which matters for ``alpha_mode="flatten"`` where
an opaque picture is the whole point.

``Image.MAX_IMAGE_PIXELS`` is lifted because the tool targets big AI exports.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
from PIL import Image

Image.MAX_IMAGE_PIXELS = None


def load_rgba(path: str | Path) -> np.ndarray:
    """Any format Pillow reads -> HxWx4 uint8."""
    image = Image.open(path)
    if image.mode != "RGBA":
        image = image.convert("RGBA")
    return np.asarray(image)


def save_rgba(rgba: np.ndarray, path: str | Path) -> Path:
    """Write an RGBA array; drop the alpha channel when it is uniformly opaque."""
    array = np.asarray(rgba)
    if array.ndim != 3 or array.shape[2] != 4:
        raise ValueError("expected an HxWx4 RGBA image")
    path = Path(path)
    if bool((array[..., 3] == 255).all()):
        Image.fromarray(array[..., :3], "RGB").save(path)
    else:
        Image.fromarray(array, "RGBA").save(path)
    return path
