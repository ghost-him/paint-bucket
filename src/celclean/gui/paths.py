"""Where the GUI keeps its settings, and how "portable" (绿色版) is decided.

Portable means: nothing is written to the registry, the settings file lives next to the executable,
so the app can be copied to a USB stick, used on someone else's machine and leaves no trace behind.

Rules, in order:

1. ``--portable`` on the command line, or ``CELCLEAN_PORTABLE=1`` in the environment: portable.
2. A packed build (PyInstaller) started from a writable directory: portable by default, because a
   downloaded single-file exe has no installer that would own a registry entry.
3. Otherwise (running from source, or a read-only directory like ``C:\\Program Files``): the
   platform-native store -- the registry on Windows.

Only the settings are affected: the images the user opens and the files the tool writes stay
wherever the user pointed them, and a packed build's temporary unpack directory is removed on exit.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

SETTINGS_NAME = "celclean.ini"
ENV_FORCE = "CELCLEAN_PORTABLE"
TRUE_VALUES = {"1", "true", "yes", "on"}
NATIVE_DESCRIPTION = "注册表 HKCU\\Software\\celclean\\celclean-gui"


def is_frozen() -> bool:
    """True inside a PyInstaller build (also true for other frozen launchers)."""
    return bool(getattr(sys, "frozen", False))


def base_dir() -> Path:
    """The directory the app treats as its own: next to the exe, or the repository root."""
    if is_frozen():
        return Path(sys.executable).resolve().parent
    return Path(__file__).resolve().parents[3]


def forced() -> bool:
    return os.environ.get(ENV_FORCE, "").strip().lower() in TRUE_VALUES


def writable(directory: Path) -> bool:
    """Can the settings file be created there? Probed, not guessed from permissions."""
    try:
        directory.mkdir(parents=True, exist_ok=True)
        with open(directory / SETTINGS_NAME, "a", encoding="utf-8"):
            pass
    except OSError:
        return False
    return True


def settings_file(force: bool = False) -> Path | None:
    """Portable settings path, or None to use the platform-native store."""
    if force or forced():
        return base_dir() / SETTINGS_NAME
    if is_frozen() and writable(base_dir()):
        return base_dir() / SETTINGS_NAME
    return None


def describe(force: bool = False) -> str:
    """One line for the About box and the self test."""
    path = settings_file(force)
    if path is None:
        return f"配置：{NATIVE_DESCRIPTION}"
    if not writable(path.parent):
        return f"配置：{path}（目录不可写，设置可能存不下来）"
    return f"配置：{path}（便携模式，不写注册表）"
