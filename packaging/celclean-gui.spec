# -*- mode: python ; coding: utf-8 -*-
"""One-file Windows build of the GUI.

    uv run --extra gui --group pack pyinstaller --noconfirm --clean packaging/celclean-gui.spec

Produces ``dist/celclean-gui.exe`` (windowed, no console).  Verify it with::

    dist/celclean-gui.exe --selftest --selftest-out selftest.json --out-dir build-check

``excludes`` trims the PySide6 modules this app never touches (WebEngine, QML/Quick, 3D,
Multimedia, SQL, ...).  Anything removed that turns out to be needed shows up immediately as a
failed ``--selftest``, so keep that check in the loop when editing this list.
"""

from pathlib import Path

ROOT = Path(SPECPATH).parent
ICON = ROOT / "packaging" / "assets" / "celclean.ico"

EXCLUDES = [
    # PySide6 modules the GUI does not use
    "PySide6.Qt3DAnimation",
    "PySide6.Qt3DCore",
    "PySide6.Qt3DExtras",
    "PySide6.Qt3DInput",
    "PySide6.Qt3DLogic",
    "PySide6.Qt3DRender",
    "PySide6.QtBluetooth",
    "PySide6.QtCharts",
    "PySide6.QtDataVisualization",
    "PySide6.QtDesigner",
    "PySide6.QtGraphs",
    "PySide6.QtHelp",
    "PySide6.QtLocation",
    "PySide6.QtMultimedia",
    "PySide6.QtMultimediaWidgets",
    "PySide6.QtNfc",
    "PySide6.QtOpenGL",
    "PySide6.QtOpenGLWidgets",
    "PySide6.QtPdf",
    "PySide6.QtPdfWidgets",
    "PySide6.QtPositioning",
    "PySide6.QtQml",
    "PySide6.QtQuick",
    "PySide6.QtQuick3D",
    "PySide6.QtQuickControls2",
    "PySide6.QtQuickWidgets",
    "PySide6.QtRemoteObjects",
    "PySide6.QtScxml",
    "PySide6.QtSensors",
    "PySide6.QtSerialPort",
    "PySide6.QtSpatialAudio",
    "PySide6.QtSql",
    "PySide6.QtStateMachine",
    "PySide6.QtSvg",
    "PySide6.QtSvgWidgets",
    "PySide6.QtTest",
    "PySide6.QtTextToSpeech",
    "PySide6.QtWebChannel",
    "PySide6.QtWebEngineCore",
    "PySide6.QtWebEngineQuick",
    "PySide6.QtWebEngineWidgets",
    "PySide6.QtWebSockets",
    # standard library / third party ballast
    "tkinter",
    "unittest",
    "pydoc_data",
    "pytest",
    "IPython",
    "matplotlib",
    "scipy",
    "pandas",
]

a = Analysis(
    [str(ROOT / "packaging" / "celclean_gui_entry.py")],
    pathex=[str(ROOT / "src")],
    binaries=[],
    datas=[(str(ICON), "assets")] if ICON.exists() else [],
    hiddenimports=[],
    hookspath=[],
    runtime_hooks=[],
    excludes=EXCLUDES,
    noarchive=False,
    optimize=0,
)
pyz = PYZ(a.pure)

exe = EXE(
    pyz,
    a.scripts,
    a.binaries,
    a.datas,
    [],
    name="celclean-gui",
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=False,
    console=False,
    disable_windowed_traceback=False,
    icon=str(ICON) if ICON.exists() else None,
)
