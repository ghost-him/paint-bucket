"""The GUI's parameter model: knobs, presets, time estimate -- no Qt, so it is unit-testable.

The values mirror the CLI defaults; `params.py` only puts widgets in front of them.
"""

from __future__ import annotations

from dataclasses import dataclass, fields

from ..pipeline import Options, parse_color

REF_PIXELS = 1254 * 1254
REF_SECONDS = 3.0  # measured wall clock of the reference image at stride 1, snap off
REF_REPORT_SECONDS = 15.0  # same image with --snap
STRIDE_2_SPEEDUP = 3.4
SIZE_EXPONENT = 1.40  # 1254² -> 3.0 s, 4096² -> 82.9 s

#: name -> field overrides; everything not listed keeps the dataclass default
PRESETS: dict[str, dict] = {
    "默认（推荐）": {},
    "更快（大图 / 粗调）": {"stride": 2},
    "更干净（慢）": {"radius": 20},
    "最纯（snap，柔和渐变上会有硬边色斑）": {"snap": True, "snap_mode": "plane"},
    "保守（厚涂 / 照片）": {"strength": 0.5, "stride": 2},
}


@dataclass
class GuiParams:
    stride: int = 1
    radius: int = 0  # 0 = auto from the image size
    strength: float = 1.0
    snap: bool = False
    snap_mode: str = "plane"
    snap_tol: float = 0.5
    min_block: int = 64
    plane_tol: float = 4.0
    slope_tol: float = 0.05
    aa_band: int = 2
    alpha: str = "normalize"
    sigma: float = 0.0  # 0 = measure it
    bg_color: str = "#ffffff"  # used by alpha == "flatten"
    suffix: str = "-clean"
    write_report: bool = False

    def to_options(self) -> Options:
        return Options(
            strength=float(self.strength),
            snap=bool(self.snap),
            snap_mode=str(self.snap_mode),
            snap_tol=float(self.snap_tol),
            min_block=int(self.min_block),
            plane_tol=float(self.plane_tol),
            slope_tol=float(self.slope_tol),
            aa_band=int(self.aa_band),
            radius=int(self.radius) or None,
            stride=int(self.stride),
            sigma_grain=float(self.sigma) or None,
            alpha_mode=str(self.alpha),
            bg_color=parse_color(self.bg_color),
        )

    def measure_background(self) -> tuple[int, int, int]:
        """The colour the output sits on: its own background when flattening, else white."""
        return parse_color(self.bg_color) if self.alpha == "flatten" else (255, 255, 255)

    def measure_composite(self) -> tuple[int, int, int] | None:
        """Composite the original for the report too? Only a flattened output needs it."""
        return parse_color(self.bg_color) if self.alpha == "flatten" else None

    def key(self) -> tuple:
        return tuple(getattr(self, f.name) for f in fields(self))

    def summary(self) -> str:
        radius = "auto" if not self.radius else str(self.radius)
        bits = [f"stride {self.stride}", f"半径 {radius}", f"strength {self.strength:g}"]
        if self.snap:
            bits.append(f"snap({self.snap_mode})")
        if self.alpha == "keep":
            bits.append("alpha 保留")
        elif self.alpha == "flatten":
            bits.append(f"背景 {self.bg_color}")
        return " · ".join(bits)


TRUE_WORDS = {"1", "true", "yes", "on"}


def _coerce(value: object, default: object) -> object:
    """Put one stored value back into the type of its default.

    QSettings hands back plain strings for an INI file, and `bool("false")` is True -- silently
    loading "false" as enabled is exactly the kind of bug that hides behind a catch-all.
    """
    if isinstance(default, bool):
        return value.strip().lower() in TRUE_WORDS if isinstance(value, str) else bool(value)
    try:
        return type(default)(value)  # type: ignore[call-arg]
    except (TypeError, ValueError):
        return default


def params_from_mapping(raw: dict, defaults: GuiParams | None = None) -> GuiParams:
    """Rebuild :class:`GuiParams` from whatever a settings store returned (unknown keys ignored)."""
    base = defaults or GuiParams()
    values = {f.name: _coerce(raw.get(f.name, getattr(base, f.name)), getattr(base, f.name)) for f in fields(base)}
    return GuiParams(**values)


def estimate_seconds(width: int, height: int, params: GuiParams) -> float:
    """Rough wall clock for one image; only meant to set expectations for large files."""
    base = REF_SECONDS * ((width * height) / REF_PIXELS) ** SIZE_EXPONENT
    if params.stride >= 2:
        base /= STRIDE_2_SPEEDUP
    if params.snap:
        base *= REF_REPORT_SECONDS / REF_SECONDS
    return base


