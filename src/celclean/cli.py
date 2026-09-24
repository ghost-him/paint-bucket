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
from .pipeline import Options, clean
from .qa import image_metrics, pick_crops, write_comparison, write_report

Image.MAX_IMAGE_PIXELS = None


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


def _load(path: str | Path) -> np.ndarray:
    img = Image.open(path)
    if img.mode != "RGBA":
        img = img.convert("RGBA")
    return np.asarray(img)


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
        aa_band=args.aa_band,
    )


def _default_output(input_path: Path, suffix: str | None = None) -> Path:
    tag = suffix or "-clean"
    return input_path.with_name(f"{input_path.stem}{tag}.png")


def cmd_clean(args: argparse.Namespace) -> int:
    src = Path(args.input)
    if not src.exists():
        raise SystemExit(f"no such file: {src}")
    out_path = Path(args.output) if args.output else _default_output(src, args.suffix)
    rgba = _load(src)
    cleaned, info = clean(rgba, _options(args))
    Image.fromarray(cleaned, "RGBA").save(out_path)
    print(f"wrote {out_path}  ({info['size'][0]}x{info['size'][1]}, radius={info['radius']}, "
          f"sigma_range={info['sigma_range']}, snap={info['snap']})")
    if args.report:
        metrics = image_metrics(rgba, cleaned)
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
    rgba = _load(src)
    if args.cleaned:
        cleaned_path = Path(args.cleaned)
        if not cleaned_path.exists():
            raise SystemExit(f"no such file: {cleaned_path}")
        cleaned = _load(cleaned_path)
    else:
        cleaned, info = clean(rgba, _options(args))
        cleaned_path = _default_output(src, args.suffix)
        Image.fromarray(cleaned, "RGBA").save(cleaned_path)
        print(f"wrote {cleaned_path} (cleaned on the fly)")

    crops = _parse_crops(args.crop)
    if crops is None and not args.no_auto_crop:
        crops = pick_crops(rgba, cleaned)
    out_dir = Path(args.out_dir) if args.out_dir else src.with_name(src.stem + "-compare")
    files = write_comparison(rgba, cleaned, out_dir, crops=crops, zoom=args.zoom)
    for f in files:
        print(f"wrote {f}")
    metrics = image_metrics(rgba, cleaned)
    write_report(metrics, {"input": str(src), "cleaned": str(cleaned_path)}, out_dir / "report.json")
    _print_metrics(metrics)
    return 0


def _print_metrics(m: dict) -> None:
    print("\n--- measured (acceptance protocol) ---")
    print(f"  flattest tiles ({len(m['flat_tiles'])}x 32x32): std {m['flat_tile_std_before_mean']} -> "
          f"{m['flat_tile_std_after_mean']} mean, {m['flat_tile_std_after_median']} median; "
          f"{m['flat_tiles_cleaned']} are now flat (std<=0.1); worst colour bias "
          f"{m['flat_tile_bias_absmax']} levels")
    print(f"  local variation on flat pixels: {m['local_variation_flat_px_before']} -> "
          f"{m['local_variation_flat_px_after']}")
    print(f"  grain sigma (levels): {m['sigma_grain_before_levels']} -> {m['sigma_grain_after_levels']}")
    print(f"  luma corr {m['corr_luma']}  mean|d| {m['mean_abs_delta_levels']}  "
          f"p99|d| {m['p99_abs_delta_levels']}  max|d| {m['max_abs_delta_levels']}")
    print(f"  pixels changed >2 levels {100 * m['frac_gt2_levels']:.2f}%  "
          f">8 levels {100 * m['frac_gt8_levels']:.2f}%")


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="celclean",
        description="Clean AI-generated cel-shaded art: remove grain/mottle, keep gradients and thin features.",
    )
    p.add_argument("--version", action="version", version=f"celclean {__version__}")
    sub = p.add_subparsers(dest="command", required=True)

    def common(sp: argparse.ArgumentParser) -> None:
        sp.add_argument("--strength", type=float, default=1.0, help="1.0 = default, higher = flatter (default: 1.0)")
        sp.add_argument("--snap", action=argparse.BooleanOptionalAction, default=False,
                        help="repaint each connected flat block with exactly one colour; gives pure "
                             "blocks but turns gentle shading into visible steps (default: off)")
        sp.add_argument("--snap-tol", type=float, default=0.5,
                        help="max distance (dE) from the local mean that still counts as flat; "
                             "lower = keep more shading (default: 0.5)")
        sp.add_argument("--min-block", type=int, default=64,
                        help="blocks smaller than this many px are left to the denoiser (default: 64)")
        sp.add_argument("--radius", type=int, default=None, help="core window radius in px (default: auto)")
        sp.add_argument("--stride", type=int, default=2, help="window sampling step; 1 = slowest/best (default: 2)")
        sp.add_argument("--sigma", type=float, default=None, help="override the measured grain sigma (Lab L units)")
        sp.add_argument("--alpha", choices=["normalize", "keep"], default="normalize",
                        help="normalize: interior alpha -> 255 (default)")
        sp.add_argument("--aa-band", type=int, default=2, help="px of edge kept un-snapped (default: 2)")

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
