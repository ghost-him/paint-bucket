"""Command line interface.

    celclean clean   head4.png                 # writes head4-clean.png
    celclean compare head4.png head4-clean.png # writes before/after figures
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
from PIL import Image

from . import __version__
from .io import load_rgba, save_rgba
from .pipeline import Options, clean, parse_color
from .qa import image_metrics, metrics_text, pick_crops, write_comparison, write_report


def _parse_crops(values: list[str] | None) -> list[tuple[int, int, int, int]] | None:
    if not values:
        return None
    crops = []
    for v in values:
        parts = [int(p) for p in v.replace(",", " ").split()]
        if len(parts) != 4:
            raise SystemExit(f"--crop expects x,y,w,h (got {v!r})")
        crops.append(tuple(parts))  # type: ignore[arg-type]
    return crops


def _color(text: str) -> tuple[int, int, int]:
    """argparse type for --bg-color: report a bad value the way argparse likes it."""
    try:
        return parse_color(text)
    except ValueError as exc:
        raise argparse.ArgumentTypeError(str(exc)) from exc


def _background(opts: Options) -> tuple[int, int, int]:
    """What to render both sides over: the output's own colour when flattening, else white."""
    return tuple(opts.bg_color) if opts.alpha_mode == "flatten" else (255, 255, 255)


def _composite_original(opts: Options) -> tuple[int, int, int] | None:
    """A flattened output is compared against a flattened original (see qa.image_metrics)."""
    return tuple(opts.bg_color) if opts.alpha_mode == "flatten" else None


def _options(args: argparse.Namespace) -> Options:
    return Options(
        strength=args.strength,
        snap=args.snap,
        snap_tol=args.snap_tol,
        min_block=args.min_block,
        radius=args.radius,
        stride=args.stride,
        sigma_grain=args.sigma,
        alpha_mode=args.alpha,
        bg_color=args.bg_color,
        aa_band=args.aa_band,
        snap_mode=args.snap_mode,
        plane_tol=args.plane_tol,
        slope_tol=args.slope_tol,
    )


def _default_output(input_path: Path, suffix: str | None = None) -> Path:
    tag = suffix or "-clean"
    return input_path.with_name(f"{input_path.stem}{tag}.png")


def cmd_clean(args: argparse.Namespace) -> int:
    src = Path(args.input)
    if not src.exists():
        raise SystemExit(f"no such file: {src}")
    out_path = Path(args.output) if args.output else _default_output(src, args.suffix)
    rgba = load_rgba(src)
    opts = _options(args)
    cleaned, info = clean(rgba, opts)
    save_rgba(cleaned, out_path)
    print(f"wrote {out_path}  ({info['size'][0]}x{info['size'][1]}, radius={info['radius']}, "
          f"sigma_range={info['sigma_range']}, snap={info['snap']})")
    if args.report:
        metrics = image_metrics(
            rgba, cleaned, background=_background(opts), composite_original=_composite_original(opts)
        )
        report_path = (
            Path(args.report) if args.report != "auto" else src.with_name(f"{src.stem}-report.json")
        )
        write_report(metrics, info, report_path)
        print(f"wrote {report_path}")
        _print_metrics(metrics)
    return 0


def cmd_compare(args: argparse.Namespace) -> int:
    src = Path(args.input)
    if not src.exists():
        raise SystemExit(f"no such file: {src}")
    rgba = load_rgba(src)
    opts = _options(args)
    if args.cleaned:
        cleaned_path = Path(args.cleaned)
        if not cleaned_path.exists():
            raise SystemExit(f"no such file: {cleaned_path}")
        cleaned = load_rgba(cleaned_path)
    else:
        cleaned, info = clean(rgba, opts)
        cleaned_path = _default_output(src, args.suffix)
        save_rgba(cleaned, cleaned_path)
        print(f"wrote {cleaned_path} (cleaned on the fly)")

    crops = _parse_crops(args.crop)
    if crops is None and not args.no_auto_crop:
        crops = pick_crops(rgba, cleaned)
    out_dir = Path(args.out_dir) if args.out_dir else src.with_name(src.stem + "-compare")
    files = write_comparison(rgba, cleaned, out_dir, crops=crops, zoom=args.zoom, background=_background(opts))
    for f in files:
        print(f"wrote {f}")
    metrics = image_metrics(
        rgba, cleaned, background=_background(opts), composite_original=_composite_original(opts)
    )
    write_report(metrics, {"input": str(src), "cleaned": str(cleaned_path)}, out_dir / "report.json")
    _print_metrics(metrics)
    return 0


def _print_metrics(m: dict) -> None:
    print(metrics_text(m))


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="celclean",
        description="Clean AI-generated cel-shaded art: remove grain/mottle, keep gradients and thin features.",
    )
    p.add_argument("--version", action="version", version=f"celclean {__version__}")
    sub = p.add_subparsers(dest="command", required=True)

    def common(sp: argparse.ArgumentParser) -> None:
        sp.add_argument("--strength", type=float, default=1.0,
                        help="scales the colour-similarity threshold of the denoiser (default: 1.0). "
                             "Measured: >1 does NOT make flat blocks cleaner, it only enlarges the "
                             "worst-case change (max |d| 7.0 -> 21.3 levels at 3.0); for purer blocks "
                             "use --stride 1 or --snap")
        sp.add_argument("--snap", action=argparse.BooleanOptionalAction, default=False,
                        help="repaint each connected flat block with its own colour instead of "
                             "averaging: purest blocks (flat std 0.227 -> 0.045, 7/8 windows flat) but "
                             "large gentle gradients pick up hard-edged patches (verified visually at "
                             "1-2 level differences), so it stays off by default")
        sp.add_argument("--snap-tol", type=float, default=0.5,
                        help="max distance (dE) from the local mean that still counts as flat; "
                             "lower = keep more shading (default: 0.5)")
        sp.add_argument("--min-block", type=int, default=64,
                        help="blocks smaller than this many px are left to the denoiser (default: 64)")
        sp.add_argument("--radius", type=int, default=None, help="core window radius in px (default: auto)")
        sp.add_argument("--stride", type=int, default=1,
                        help="window sampling step (default: 1). 1 = every tap: the measured quality "
                             "work point (flat-pixel variation 0.0548 vs 0.0807, uniform 3x3 "
                             "neighbourhoods 0.554 vs 0.464, worst-case change 7.0 -> 8.0 levels) at "
                             "~3.4x the time; use 2 to trade purity back for speed")
        sp.add_argument("--sigma", type=float, default=None, help="override the measured grain sigma (Lab L units)")
        sp.add_argument("--alpha", choices=["normalize", "keep", "flatten"], default="normalize",
                        help="normalize: interior alpha -> 255 (default). This changes more pixels "
                             "than the denoising itself (253 -> 255 over 77%% of the image); 'keep' "
                             "leaves alpha untouched; 'flatten' also composites the whole image onto "
                             "one solid colour (--bg-color) and drops the alpha channel, so the "
                             "output is an opaque RGB PNG")
        sp.add_argument("--bg-color", type=_color, default=(255, 255, 255), metavar="#RRGGBB",
                        help="background for --alpha flatten (default: #ffffff; also accepts "
                             "'white'/'black' or R,G,B); ignored by the other alpha modes")
        sp.add_argument("--aa-band", type=int, default=2,
                        help="px of edge kept un-snapped (default: 2); has an effect only together "
                             "with --snap (with snap off the output is byte-identical)")
        sp.add_argument("--snap-mode", choices=["plane", "constant"], default="plane",
                        help="with --snap: 'plane' fits a plane per block (follows gentle ramps, so "
                             "no staircase) while 'constant' paints one colour per block (default: plane)")
        sp.add_argument("--plane-tol", type=float, default=4.0,
                        help="with --snap-mode plane: keep a block if its plane residual is at most "
                             "this many times the median block residual (floor 0.5 dE); blocks that "
                             "fit far worse (curved shading, leaked edges) are left to the denoiser "
                             "(default: 4.0)")
        sp.add_argument("--slope-tol", type=float, default=0.05,
                        help="max slope difference (dE/px) for two adjacent blocks to be merged in "
                             "--snap-mode plane (default: 0.05)")

    c = sub.add_parser("clean", help="clean an image")
    c.add_argument("input")
    c.add_argument("-o", "--output", default=None)
    c.add_argument("--suffix", default=None, help="output name suffix (default: -clean)")
    c.add_argument("--report", nargs="?", const="auto", default=None,
                   help="write a JSON report of the measured numbers (optionally: path)")
    common(c)
    c.set_defaults(func=cmd_clean)

    m = sub.add_parser("compare", help="write before/after figures")
    m.add_argument("input")
    m.add_argument("cleaned", nargs="?", default=None, help="omit to clean on the fly")
    m.add_argument("--suffix", default=None)
    m.add_argument("--out-dir", default=None)
    m.add_argument("--crop", action="append", default=None, metavar="X,Y,W,H",
                   help="detail crop; repeatable (default: two automatic crops)")
    m.add_argument("--no-auto-crop", action="store_true")
    m.add_argument("--zoom", type=int, default=3)
    common(m)
    m.set_defaults(func=cmd_compare)
    return p


def main(argv: list[str] | None = None) -> int:
    argv = list(sys.argv[1:] if argv is None else argv)
    parser = build_parser()
    args = parser.parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
