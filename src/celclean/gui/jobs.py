"""Background jobs: one image, a batch of images, or the measurement pass.

Qt rules kept here: the pipeline runs on a worker thread (:class:`QRunnable`), results travel back
as signals, and every message carries a sequence number so a stale run (the user moved a slider
while it was working) can be dropped instead of overwriting a newer preview.
"""

from __future__ import annotations

import time
import traceback
from pathlib import Path
from threading import Event

import numpy as np
from PySide6.QtCore import QObject, QRunnable, Signal

from ..io import load_rgba, save_rgba
from ..pipeline import Options, clean
from ..qa import image_metrics, write_report

IMAGE_SUFFIXES = {".png", ".jpg", ".jpeg", ".webp", ".bmp", ".tif", ".tiff", ".ppm", ".gif"}


def output_path(source: Path, suffix: str) -> Path:
    return source.with_name(f"{source.stem}{suffix or '-clean'}.png")


class CleanSignals(QObject):
    done = Signal(int, object, dict, float)  # seq, rgba, info, seconds
    failed = Signal(int, str)


class CleanJob(QRunnable):
    def __init__(self, seq: int, rgba: np.ndarray, opts: Options, signals: CleanSignals) -> None:
        super().__init__()
        self.seq = seq
        self._rgba = rgba
        self._opts = opts
        self._signals = signals

    def run(self) -> None:
        started = time.perf_counter()
        try:
            out, info = clean(self._rgba, self._opts)
        except Exception:
            self._signals.failed.emit(self.seq, traceback.format_exc())
            return
        self._signals.done.emit(self.seq, out, info, time.perf_counter() - started)


class MetricSignals(QObject):
    done = Signal(object)  # metrics dict
    failed = Signal(str)


class MetricJob(QRunnable):
    """The acceptance numbers of one before/after pair (a couple of seconds of numpy)."""

    def __init__(
        self,
        original: np.ndarray,
        cleaned: np.ndarray,
        signals: MetricSignals,
        background: tuple[int, int, int] = (255, 255, 255),
        composite_original: tuple[int, int, int] | None = None,
        report: tuple[Path, dict] | None = None,
    ) -> None:
        super().__init__()
        self._original = original
        self._cleaned = cleaned
        self._signals = signals
        self._background = tuple(int(v) for v in background)
        self._composite = None if composite_original is None else tuple(int(v) for v in composite_original)
        # (path, info): written here rather than in the GUI thread, so a report survives the user
        # closing the window right after saving
        self._report = report

    def run(self) -> None:
        try:
            metrics = image_metrics(
                self._original,
                self._cleaned,
                background=self._background,
                composite_original=self._composite,
            )
            if self._report is not None:
                write_report(metrics, self._report[1], self._report[0])
        except Exception:
            self._signals.failed.emit(traceback.format_exc())
            return
        self._signals.done.emit(metrics)


class BatchSignals(QObject):
    progress = Signal(int, int, str)  # index, total, current file
    fileDone = Signal(str, bool, str)  # path, ok, message
    finished = Signal(int, int, list)  # cleaned, failed, failure messages


class BatchJob(QRunnable):
    """Clean a list of files with the same parameters.

    Cancellation is checked between files: a running image cannot be interrupted (the pipeline has
    no progress hook), so the last file always completes.
    """

    def __init__(
        self,
        paths: list[Path],
        params,
        signals: BatchSignals,
        cancel: Event,
        report: bool = False,
        overwrite: bool = True,
        out_dir: Path | None = None,
    ) -> None:
        super().__init__()
        self._paths = list(paths)
        self._params = params
        self._signals = signals
        self._cancel = cancel
        self._report = report
        self._overwrite = overwrite
        self._out_dir = Path(out_dir) if out_dir else None

    def _target(self, path: Path) -> Path:
        name = f"{path.stem}{self._params.suffix or '-clean'}.png"
        return (self._out_dir / name) if self._out_dir else path.with_name(name)

    def run(self) -> None:
        opts: Options = self._params.to_options()
        total = len(self._paths)
        cleaned = 0
        failures: list[str] = []
        if self._out_dir is not None:
            self._out_dir.mkdir(parents=True, exist_ok=True)
        for index, path in enumerate(self._paths):
            if self._cancel.is_set():
                break
            self._signals.progress.emit(index, total, str(path))
            target = self._target(path)
            if not self._overwrite and target.exists():
                self._signals.fileDone.emit(str(path), True, "skipped (exists)")
                continue
            try:
                rgba = load_rgba(path)
                out, info = clean(rgba, opts)
                save_rgba(out, target)
                if self._report:
                    write_report(
                        image_metrics(
                            rgba,
                            out,
                            background=self._params.measure_background(),
                            composite_original=self._params.measure_composite(),
                        ),
                        info,
                        target.with_name(target.stem + "-report.json"),
                    )
            except Exception as exc:  # one bad file must not kill the batch
                failures.append(f"{path.name}: {type(exc).__name__}: {exc}")
                self._signals.fileDone.emit(str(path), False, str(exc))
                continue
            cleaned += 1
            self._signals.fileDone.emit(str(path), True, "")
        self._signals.progress.emit(total, total, "")
        self._signals.finished.emit(cleaned, len(failures), failures)
