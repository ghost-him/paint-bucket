"""Draw the application icon (``packaging/assets/celclean.ico``).

The motif is the tool itself: one flat block that is dirty on the left and clean on the right.
Run it from the repository root::

    uv run --extra gui packaging/make_icon.py
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw

SIZES = (16, 24, 32, 48, 64, 128, 256)
CANVAS = 1024  # drawn big, then reduced per size
OUT = Path(__file__).resolve().parent / "assets" / "celclean.ico"
BACKGROUND = (43, 46, 58, 255)
DIRTY = (168, 176, 196, 255)
CLEAN = (238, 241, 248, 255)
RADIUS = 180


def build() -> Image.Image:
    canvas = Image.new("RGBA", (CANVAS, CANVAS), (0, 0, 0, 0))
    draw = ImageDraw.Draw(canvas)
    draw.rounded_rectangle([0, 0, CANVAS - 1, CANVAS - 1], RADIUS, fill=BACKGROUND)

    # the block, inset from the rounded background
    pad = 150
    left, top, right, bottom = pad, pad, CANVAS - pad, CANVAS - pad
    half = (left + right) // 2

    patch = np.zeros((bottom - top, right - left, 4), dtype=np.uint8)
    rng = np.random.default_rng(7)
    grain = rng.normal(0.0, 9.0, size=(patch.shape[0], patch.shape[1]))
    for channel in range(3):
        plane = np.clip(DIRTY[channel] + grain, 0, 255).astype(np.uint8)
        patch[:, :, channel] = plane
    patch[:, :, 3] = 255
    patch[:, half - left :, :3] = CLEAN[:3]  # right half: cleaned
    canvas.alpha_composite(Image.fromarray(patch, "RGBA"), (left, top))

    # a hairline between the two halves, so the motif survives the 16 px reduction
    width = max(6, (right - left) // 60)
    draw.rectangle([half - width // 2, top, half + width // 2, bottom], fill=BACKGROUND)
    return canvas


def main() -> None:
    icon = build()
    OUT.parent.mkdir(parents=True, exist_ok=True)
    icon.save(OUT, format="ICO", sizes=[(s, s) for s in SIZES])
    print(f"wrote {OUT} ({OUT.stat().st_size} bytes) with sizes {SIZES}")


if __name__ == "__main__":
    main()
