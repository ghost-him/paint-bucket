"""Qt (PySide6) desktop front-end for :mod:`celclean`.

Optional: the package works head-less, and nothing here is imported until the GUI is started.
Install the extra first::

    uv sync --extra gui            # in this repository
    pip install "celclean[gui]"    # as a dependency
"""

from __future__ import annotations


def main(argv: list[str] | None = None) -> int:
    """Entry point of the ``celclean-gui`` script; imports Qt only when actually starting."""
    try:
        from .app import main as run
    except ImportError as exc:  # PySide6 missing: say what to install instead of a traceback
        raise SystemExit(
            "celclean-gui 需要 PySide6。请先安装可选依赖：\n"
            '    uv sync --extra gui        （本仓库）\n'
            '    pip install "celclean[gui]"（作为依赖）\n'
            f"原始错误：{exc}"
        ) from exc
    return run(argv)


__all__ = ["main"]
