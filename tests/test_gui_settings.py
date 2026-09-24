"""The GUI's parameter model (Qt-free): defaults must mirror the CLI, presets must do what they say."""

from celclean.gui.settings import PRESETS, GuiParams, estimate_seconds, params_from_mapping


def test_defaults_mirror_the_cli() -> None:
    options = GuiParams().to_options()
    assert (options.stride, options.radius, options.snap, options.aa_band) == (1, None, False, 2)
    assert options.alpha_mode == "normalize" and options.sigma_grain is None
    assert options.strength == 1.0 and options.snap_mode == "plane"


def test_flatten_mode_carries_the_chosen_colour() -> None:
    params = GuiParams(alpha="flatten", bg_color="#101820")
    assert params.to_options().bg_color == (16, 24, 32)
    assert params.measure_background() == (16, 24, 32)
    assert "#101820" in params.summary()


def test_the_page_colour_is_white_unless_flattening() -> None:
    # the metrics render both sides on the colour the output actually has
    assert GuiParams(alpha="normalize", bg_color="#101820").measure_background() == (255, 255, 255)
    assert GuiParams(alpha="keep").measure_background() == (255, 255, 255)


def test_a_preset_keeps_the_other_defaults() -> None:
    params = GuiParams(**PRESETS["更干净（慢）"])
    assert params.radius == 20 and params.stride == 1 and not params.snap
    conservative = GuiParams(**PRESETS["保守（厚涂 / 照片）"])
    assert conservative.strength == 0.5 and conservative.stride == 2


def test_the_time_estimate_follows_the_measured_work_points() -> None:
    default = GuiParams()
    fast = GuiParams(stride=2)
    assert abs(estimate_seconds(1254, 1254, default) - 3.0) < 0.1  # the reference measurement
    assert 70 <= estimate_seconds(4096, 4096, default) <= 95  # measured 82.9 s
    assert estimate_seconds(4096, 4096, fast) < estimate_seconds(4096, 4096, default) / 3
    assert estimate_seconds(1254, 1254, GuiParams(snap=True)) > 4 * estimate_seconds(1254, 1254, default)


def test_params_survive_a_string_only_store() -> None:
    """An INI store hands back strings, and bool("false") is True -- hence the per-field coercion."""
    raw = {
        "stride": "2",
        "radius": "20",
        "strength": "0.5",
        "snap": "true",
        "write_report": "false",
        "snap_tol": "0.25",
        "alpha": "flatten",
        "bg_color": "#101820",
    }
    params = params_from_mapping(raw)
    assert (params.stride, params.radius, params.snap_tol) == (2, 20, 0.25)
    assert params.snap is True
    assert params.write_report is False  # the trap: bool("false") would be True
    assert params.alpha == "flatten" and params.bg_color == "#101820"
    # untouched fields keep their defaults
    assert params.min_block == GuiParams().min_block and params.suffix == "-clean"


def test_params_from_mapping_ignores_junk_and_unknown_keys() -> None:
    params = params_from_mapping({"stride": "abc", "radius": None, "nonsense": 1, "snap": None})
    assert (params.stride, params.radius) == (1, 0)
    assert params.snap is False
