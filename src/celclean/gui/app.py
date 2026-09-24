"""Main window and entry point.

    celclean-gui                     open the window
    celclean-gui picture.png         open a file (also what a file-association launch does)
    celclean-gui --selftest          head-less check, used to verify a packaged build

Keys: Ctrl+O open, Ctrl+B batch, Ctrl+S save, Ctrl+Shift+S save as, Ctrl+R clean again,
Ctrl+0 fit, Ctrl+1 1:1, Ctrl+2/3/4 side by side / original / cleaned.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path
from threading import Event

import numpy as np
from PySide6.QtCore import QEventLoop, QSettings, Qt, QThreadPool, QTimer
from PySide6.QtGui import QAction, QActionGroup, QIcon, QKeySequence
from PySide6.QtWidgets import (
    QApplication,
    QCheckBox,
    QDialog,
    QDialogButtonBox,
    QDockWidget,
    QFileDialog,
    QLabel,
    QMainWindow,
    QMessageBox,
    QPlainTextEdit,
    QProgressBar,
    QPushButton,
    QScrollArea,
    QSlider,
    QVBoxLayout,
    QWidget,
)

from .. import __version__
from ..qa import metrics_text
from .batch import BatchDialog
from .jobs import (
    IMAGE_SUFFIXES,
    BatchJob,
    BatchSignals,
    CleanJob,
    CleanSignals,
    MetricJob,
    MetricSignals,
    load_rgba,
    output_path,
    save_rgba,
)
from .params import ParamsPanel
from .settings import GuiParams, estimate_seconds
from .paths import describe as describe_settings, settings_file
from .view import GAIN_MAX, GAIN_MIN, CompareView

APP_NAME = "celclean"
ORG_NAME = "celclean"
IMAGE_FILTER = "图片 (*.png *.jpg *.jpeg *.webp *.bmp *.tif *.tiff *.gif);;所有文件 (*)"
SLOW_HINT_SECONDS = 90.0


def icon_path() -> Path | None:
    """The window/exe icon, bundled next to the sources or into a frozen build."""
    candidates = []
    bundle = getattr(sys, "_MEIPASS", None)
    if bundle:
        candidates.append(Path(bundle) / "assets" / "celclean.ico")
    candidates.append(Path(__file__).resolve().parents[3] / "packaging" / "assets" / "celclean.ico")
    return next((p for p in candidates if p.exists()), None)


class TextDialog(QDialog):
    """Read-only text with a copy button (the acceptance numbers)."""

    def __init__(self, title: str, text: str, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.setWindowTitle(title)
        self.resize(760, 480)
        view = QPlainTextEdit(text)
        view.setReadOnly(True)
        view.setLineWrapMode(QPlainTextEdit.LineWrapMode.NoWrap)
        if view.document().size().height() > 20:
            view.setLineWrapMode(QPlainTextEdit.LineWrapMode.WidgetWidth)
        copy = QPushButton("复制")
        copy.clicked.connect(lambda: QApplication.clipboard().setText(text))
        close = QPushButton("关闭")
        close.clicked.connect(self.accept)
        buttons = QDialogButtonBox()
        buttons.addButton(copy, QDialogButtonBox.ButtonRole.ActionRole)
        buttons.addButton(close, QDialogButtonBox.ButtonRole.AcceptRole)
        buttons.accepted.connect(self.accept)
        layout = QVBoxLayout(self)
        layout.addWidget(view)
        layout.addWidget(buttons)


class MainWindow(QMainWindow):
    def __init__(self, portable: bool = False) -> None:
        super().__init__()
        self.portable = portable
        self.setWindowTitle(f"{APP_NAME} — AI 赛璐璐图去噪")
        self.setAcceptDrops(True)
        self.resize(1480, 940)

        self.pool = QThreadPool.globalInstance()
        self.signals = CleanSignals()
        self.signals.done.connect(self._on_clean_done)
        self.signals.failed.connect(self._on_clean_failed)
        self.metric_signals = MetricSignals()
        self.metric_signals.done.connect(self._on_metrics)
        self.metric_signals.failed.connect(self._on_metric_failed)
        self.batch_signals = BatchSignals()
        self.batch_signals.progress.connect(self._on_batch_progress)
        self.batch_signals.finished.connect(self._on_batch_finished)

        self.rgba: np.ndarray | None = None
        self.out: np.ndarray | None = None
        self.info: dict = {}
        self.path: Path | None = None
        self.seq = 0
        self.latest = -1
        self.running = False
        self.pending: GuiParams | None = None
        self.started_at = 0.0
        self.metric_purpose = ""
        self.report_target: Path | None = None
        self.batch_cancel: Event | None = None
        self._gain_auto_done = False

        self._build_ui()
        self._build_actions()
        self._restore_settings()

        self.debounce = QTimer(self)
        self.debounce.setSingleShot(True)
        self.debounce.setInterval(600)
        self.debounce.timeout.connect(self.run_clean)
        self.ticker = QTimer(self)
        self.ticker.setInterval(250)
        self.ticker.timeout.connect(self._tick)
        self._refresh_status()

    # ---- construction --------------------------------------------------------------------
    def _build_ui(self) -> None:
        self.view = CompareView()
        self.view.cleaned.setToolTip("滚轮放大到 100% 以上，配合工具栏的「对比拉伸」查看噪点是否被洗掉")
        self.view.original.setToolTip("原图；与右侧同步缩放/平移")
        self.setCentralWidget(self.view)

        self.panel = ParamsPanel()
        self.panel.changed.connect(self._on_params_changed)
        scroll = QScrollArea()
        scroll.setWidget(self.panel)
        scroll.setWidgetResizable(True)
        scroll.setMinimumWidth(336)
        dock = QDockWidget("参数", self)
        dock.setObjectName("params-dock")
        dock.setWidget(scroll)
        dock.setAllowedAreas(Qt.DockWidgetArea.LeftDockWidgetArea | Qt.DockWidgetArea.RightDockWidgetArea)
        self.addDockWidget(Qt.DockWidgetArea.LeftDockWidgetArea, dock)
        self.params_dock = dock

        self.state_label = QLabel("拖入一张 AI 出的平涂图，或用「打开图片」")
        self.size_label = QLabel("")
        self.progress = QProgressBar()
        self.progress.setRange(0, 100)
        self.progress.setValue(0)
        self.progress.setMaximumWidth(170)
        bar = self.statusBar()
        bar.addWidget(self.state_label, 1)
        bar.addPermanentWidget(self.size_label)
        bar.addPermanentWidget(self.progress)

    def _build_actions(self) -> None:
        toolbar = self.addToolBar("主工具栏")
        toolbar.setObjectName("main-toolbar")
        toolbar.setMovable(False)

        def action(text: str, slot, shortcut: str | None = None) -> QAction:
            act = QAction(text, self)
            if shortcut:
                act.setShortcut(shortcut)
            act.triggered.connect(slot)
            return act

        self.act_open = action("打开图片", self.open_dialog, QKeySequence.StandardKey.Open)
        self.act_batch = action("批处理…", lambda: self.batch_dialog(), "Ctrl+B")
        self.act_save = action("保存结果", lambda: self.save_result(False), QKeySequence.StandardKey.Save)
        self.act_save_as = action("结果另存为…", lambda: self.save_result(True), "Ctrl+Shift+S")
        self.act_run = action("重新清理", self.run_clean, "Ctrl+R")
        self.act_fit = action("适应窗口", self.view.fit, "Ctrl+0")
        self.act_one = action("1:1", self.view.zoom_reset, "Ctrl+1")
        self.act_measure = action("测量指标（原图 vs 清理后）", self.measure)

        toolbar.addAction(self.act_open)
        toolbar.addAction(self.act_batch)
        toolbar.addSeparator()
        toolbar.addAction(self.act_save)
        toolbar.addSeparator()
        toolbar.addAction(self.act_run)
        self.auto_preview = QCheckBox("自动预览")
        self.auto_preview.setChecked(True)
        self.auto_preview.setToolTip("参数改动后自动重跑一次（单张 1254² 约 3 s）。关掉后用「重新清理」手动触发。")
        toolbar.addWidget(self.auto_preview)
        toolbar.addSeparator()
        self.detail_box = QCheckBox("噪点放大")
        self.detail_box.setChecked(False)
        self.detail_box.setToolTip(
            "去掉局部平均、按局部起伏放大：平坦区的噪点变成明显颗粒，线稿变成黑白线条。\n"
            "用来判断色块到底洗干净没有。只影响预览，不写进保存的文件。"
        )
        self.detail_box.toggled.connect(self._on_detail_toggled)
        toolbar.addWidget(self.detail_box)
        self.gain_slider = QSlider(Qt.Orientation.Horizontal)
        self.gain_slider.setRange(GAIN_MIN, GAIN_MAX)
        self.gain_slider.setValue(GAIN_MIN)
        self.gain_slider.setFixedWidth(130)
        self.gain_slider.setEnabled(False)
        self.gain_slider.valueChanged.connect(self._on_gain_changed)
        toolbar.addWidget(self.gain_slider)
        self.gain_label = QLabel("—")
        self.gain_label.setFixedWidth(44)
        toolbar.addWidget(self.gain_label)
        toolbar.addSeparator()
        toolbar.addAction(self.act_fit)
        toolbar.addAction(self.act_one)
        self.sync_box = QCheckBox("同步缩放")
        self.sync_box.setChecked(True)
        self.sync_box.toggled.connect(self.view.set_sync)
        toolbar.addWidget(self.sync_box)

        file_menu = self.menuBar().addMenu("文件")
        file_menu.addAction(self.act_open)
        file_menu.addAction(self.act_batch)
        file_menu.addSeparator()
        file_menu.addAction(self.act_save)
        file_menu.addAction(self.act_save_as)
        file_menu.addSeparator()
        file_menu.addAction(action("退出", self.close, "Ctrl+Q"))

        view_menu = self.menuBar().addMenu("视图")
        view_menu.addAction(self.act_fit)
        view_menu.addAction(self.act_one)
        view_menu.addSeparator()
        show_group = QActionGroup(self)
        for text, mode, key in (("并排", "both", "Ctrl+2"), ("仅原图", "original", "Ctrl+3"), ("仅清理后", "cleaned", "Ctrl+4")):
            act = action(text, lambda checked=False, m=mode: self._set_mode(m), key)
            act.setCheckable(True)
            act.setChecked(mode == "both")
            show_group.addAction(act)
            view_menu.addAction(act)
        view_menu.addSeparator()
        bg_group = QActionGroup(self)
        for text, mode in (("棋盘背景（看透明）", "checker"), ("白底", "white"), ("黑底", "black")):
            act = action(text, lambda checked=False, m=mode: self.view.set_background(m))
            act.setCheckable(True)
            act.setChecked(mode == "checker")
            bg_group.addAction(act)
            view_menu.addAction(act)

        tools_menu = self.menuBar().addMenu("工具")
        tools_menu.addAction(self.act_measure)
        tools_menu.addAction(action("恢复默认参数", self.panel.reset))
        tools_menu.addAction(action("参数面板", self._toggle_dock))
        about_menu = self.menuBar().addMenu("帮助")
        about_menu.addAction(action("关于", self.about))

        self._set_actions_enabled(False)

    def _toggle_dock(self) -> None:
        self.params_dock.setVisible(not self.params_dock.isVisible())

    def _set_mode(self, mode: str) -> None:
        self.view.set_mode(mode)
        self._refresh_status()

    def _set_actions_enabled(self, on: bool) -> None:
        for act in (self.act_save, self.act_save_as, self.act_measure, self.act_run):
            act.setEnabled(on)

    # ---- persistence ---------------------------------------------------------------------
    def _settings(self) -> QSettings:
        """Portable builds keep an ini next to the exe; otherwise Qt's native store is used."""
        target = settings_file(self.portable)
        if target is None:
            return QSettings(ORG_NAME, "celclean-gui")
        return QSettings(str(target), QSettings.Format.IniFormat)

    def _restore_settings(self) -> None:
        settings = self._settings()
        geometry = settings.value("window/geometry")
        if geometry is not None:
            self.restoreGeometry(geometry)
        state = settings.value("window/state")
        if state is not None:
            self.restoreState(state)
        self.panel.load(settings)

    def _save_settings(self) -> None:
        settings = self._settings()
        settings.setValue("window/geometry", self.saveGeometry())
        settings.setValue("window/state", self.saveState())
        self.panel.save(settings)
        settings.sync()  # the ini has to be on disk before the process goes away

    def closeEvent(self, event) -> None:  # noqa: N802
        self._save_settings()
        if self.batch_cancel is not None:
            self.batch_cancel.set()
        super().closeEvent(event)

    # ---- files ---------------------------------------------------------------------------
    def open_dialog(self) -> None:
        chosen, _ = QFileDialog.getOpenFileName(self, "打开图片", str(self.path.parent if self.path else ""), IMAGE_FILTER)
        if chosen:
            self.open_image(Path(chosen))

    def open_image(self, path: Path) -> None:
        try:
            rgba = load_rgba(path)
        except Exception as exc:
            QMessageBox.critical(self, APP_NAME, f"读不了这张图：\n{exc}")
            return
        self.rgba = rgba
        self.out = None
        self.info = {}
        self.path = Path(path)
        self.view.set_original(rgba)
        self.view.set_cleaned(None)
        self._gain_auto_done = False
        self._set_actions_enabled(False)
        self.act_run.setEnabled(True)
        self._apply_gain()
        self.setWindowTitle(f"{APP_NAME} — {self.path.name}")
        self._refresh_status()
        if self.auto_preview.isChecked():
            self.run_clean()
        else:
            self.state_label.setText("已载入；滚轮放大到 100% 以上，配合「对比拉伸」看噪点是否被洗掉")

    def save_result(self, ask: bool) -> None:
        if self.out is None or self.path is None:
            QMessageBox.information(self, APP_NAME, "还没有可保存的结果。")
            return
        target = output_path(self.path, self.panel.params().suffix)
        if ask:
            chosen, _ = QFileDialog.getSaveFileName(self, "保存清理结果", str(target), "PNG (*.png)")
            if not chosen:
                return
            target = Path(chosen)
        elif target.exists():
            answer = QMessageBox.question(self, APP_NAME, f"{target.name} 已存在，覆盖吗？")
            if answer != QMessageBox.StandardButton.Yes:
                return
        try:
            self._save_to(target)
        except Exception as exc:
            QMessageBox.critical(self, APP_NAME, f"写不出去：\n{exc}")

    def _save_to(self, target: Path) -> None:
        """Write the cleaned image; start the JSON report if the panel asks for one."""
        if self.out is None:
            return
        save_rgba(self.out, target)
        self.state_label.setText(f"已保存 {target}")
        if self.panel.params().write_report and self.rgba is not None:
            self.metric_purpose = "report"
            self.report_target = target.with_name(target.stem + "-report.json")
            self.state_label.setText(f"已保存 {target}，正在计算报告…")
            self.pool.start(
                MetricJob(
                    self.rgba,
                    self.out,
                    self.metric_signals,
                    self.panel.params().measure_background(),
                    self.panel.params().measure_composite(),
                    report=(self.report_target, self.info),
                )
            )

    # ---- cleaning ------------------------------------------------------------------------
    def _on_params_changed(self) -> None:
        self._refresh_status()
        if self.auto_preview.isChecked() and self.rgba is not None:
            self.debounce.start()

    def run_clean(self) -> None:
        if self.debounce.isActive():
            self.debounce.stop()
        if self.rgba is None:
            self.state_label.setText("先打开一张图片")
            return
        params = self.panel.params()
        if self.running:
            self.pending = params
            return
        self.running = True
        self.seq += 1
        self.latest = self.seq
        self.started_at = time.perf_counter()
        self.progress.setRange(0, 0)
        self.ticker.start()
        self._tick()
        self.view.cleaned.set_label("清理后（计算中…）")
        self.pool.start(CleanJob(self.seq, self.rgba, params.to_options(), self.signals))

    def _tick(self) -> None:
        elapsed = time.perf_counter() - self.started_at
        self.state_label.setText(f"清理中… 已用 {elapsed:.1f} s")

    def _on_clean_done(self, seq: int, out: np.ndarray, info: dict, seconds: float) -> None:
        if seq != self.latest:
            return
        self.running = False
        self.ticker.stop()
        self.progress.setRange(0, 100)
        self.progress.setValue(100)
        self.out = out
        self.info = info
        self.view.set_cleaned(out)
        self.view.cleaned.set_label("清理后")
        self._set_actions_enabled(True)
        self.state_label.setText(
            f"完成：{seconds:.2f} s · 半径 {info.get('radius')} · σ_range {info.get('sigma_range')} · "
            f"snap {'开' if info.get('snap') else '关'}"
        )
        self._refresh_status()
        if self.pending is not None:
            self.pending = None
            self.run_clean()

    def _on_clean_failed(self, seq: int, traceback_text: str) -> None:
        if seq != self.latest:
            return
        self.running = False
        self.ticker.stop()
        self.progress.setRange(0, 100)
        self.progress.setValue(0)
        last = [line for line in traceback_text.strip().splitlines() if line.strip()]
        self.state_label.setText("清理失败")
        QMessageBox.critical(self, APP_NAME, f"清理失败：\n{last[-1] if last else '未知错误'}")

    # ---- appearance ----------------------------------------------------------------------
    def _on_detail_toggled(self, on: bool) -> None:
        self.gain_slider.setEnabled(on)
        if on:
            self._auto_gain_once()
        self._apply_gain()

    def _auto_gain_once(self) -> None:
        """First switch-on per image: measure the flat-block texture and set the slider to it."""
        if self._gain_auto_done:
            return
        self._gain_auto_done = True
        QApplication.setOverrideCursor(Qt.CursorShape.WaitCursor)
        try:
            gain = self.view.auto_gain()
        finally:
            QApplication.restoreOverrideCursor()
        self.gain_slider.setValue(int(round(gain)))

    def _on_gain_changed(self, _value: int) -> None:
        if self.detail_box.isChecked():
            self._apply_gain()

    def _apply_gain(self) -> None:
        if not self.detail_box.isChecked():
            self.gain_label.setText("—")
            self.view.set_gain(1.0)
            return
        gain = float(self.gain_slider.value())
        self.gain_label.setText(f"{gain:g}×")
        if self.view.needs_field():  # first switch-on filters each image once (slow on 4K)
            QApplication.setOverrideCursor(Qt.CursorShape.WaitCursor)
            try:
                self.view.set_gain(gain)
            finally:
                QApplication.restoreOverrideCursor()
        else:
            self.view.set_gain(gain)

    def _refresh_status(self) -> None:
        if self.rgba is None:
            self.size_label.setText("")
            return
        height, width = self.rgba.shape[:2]
        seconds = estimate_seconds(width, height, self.panel.params())
        text = f"{width}×{height} · 预计 ~{seconds:.1f} s · {self.panel.params().summary()}"
        if seconds > SLOW_HINT_SECONDS and self.panel.params().stride == 1:
            text += "  ← 大图：采样步长改 2 可快 3.4 倍"
        self.size_label.setText(text)

    # ---- batch ---------------------------------------------------------------------------
    def batch_dialog(self, paths: list[Path] | None = None, folder: Path | None = None) -> None:
        dialog = BatchDialog(self, self.panel.params(), paths=paths, folder=folder)
        if dialog.exec() != QDialog.DialogCode.Accepted:
            return
        request = dialog.request()
        if not request.paths:
            return
        self.batch_cancel = Event()
        total = len(request.paths)
        self.progress.setRange(0, total)
        self.progress.setValue(0)
        self.state_label.setText(f"批处理 0/{total}…")
        job = BatchJob(
            request.paths,
            self.panel.params(),
            self.batch_signals,
            self.batch_cancel,
            report=request.report,
            overwrite=request.overwrite,
            out_dir=request.out_dir,
        )
        self.pool.start(job)

    def _on_batch_progress(self, index: int, total: int, current: str) -> None:
        self.progress.setValue(index)
        if current:
            self.state_label.setText(f"批处理 {index + 1}/{total}：{Path(current).name}")

    def _on_batch_finished(self, cleaned: int, failed: int, failures: list) -> None:
        self.batch_cancel = None
        self.progress.setValue(self.progress.maximum())
        self.state_label.setText(f"批处理结束：成功 {cleaned}，失败 {failed}")
        message = f"成功 {cleaned} 个，失败 {failed} 个。"
        if failures:
            shown = "\n".join(failures[:12])
            more = f"\n… 还有 {len(failures) - 12} 条" if len(failures) > 12 else ""
            message += f"\n\n失败的：\n{shown}{more}"
        QMessageBox.information(self, "批处理结束", message)

    # ---- metrics -------------------------------------------------------------------------
    def measure(self) -> None:
        if self.out is None or self.rgba is None:
            QMessageBox.information(self, APP_NAME, "先清理一张图，再测量。")
            return
        self.metric_purpose = "dialog"
        self.progress.setRange(0, 0)
        self.state_label.setText("正在测量…")
        self.pool.start(
                MetricJob(
                    self.rgba,
                    self.out,
                    self.metric_signals,
                    self.panel.params().measure_background(),
                    self.panel.params().measure_composite(),
                )
            )

    def _on_metrics(self, metrics: dict) -> None:
        self.progress.setRange(0, 100)
        self.progress.setValue(100)
        if self.metric_purpose == "report" and self.report_target is not None:
            self.state_label.setText(f"报告已写入 {self.report_target}")  # the job wrote it
        else:
            self.state_label.setText("测量完成")
            TextDialog("验收指标", metrics_text(metrics), self).exec()
        self.metric_purpose = ""
        self.report_target = None

    def _on_metric_failed(self, traceback_text: str) -> None:
        self.progress.setRange(0, 100)
        last = [line for line in traceback_text.strip().splitlines() if line.strip()]
        QMessageBox.warning(self, APP_NAME, f"测量失败：\n{last[-1] if last else '未知错误'}")

    # ---- drag and drop -------------------------------------------------------------------
    def dragEnterEvent(self, event) -> None:  # noqa: N802
        if event.mimeData().hasUrls():
            event.acceptProposedAction()

    def dropEvent(self, event) -> None:  # noqa: N802
        paths = [Path(url.toLocalFile()) for url in event.mimeData().urls() if url.isLocalFile()]
        files = [p for p in paths if p.is_file() and p.suffix.lower() in IMAGE_SUFFIXES]
        folders = [p for p in paths if p.is_dir()]
        if len(paths) == 1 and len(files) == 1:
            self.open_image(files[0])
            return
        if not files and not folders:
            self.state_label.setText("拖进来的东西里没有图片")
            return
        self.batch_dialog(paths=files, folder=folders[0] if folders else None)

    def about(self) -> None:
        QMessageBox.information(
            self,
            f"关于 {APP_NAME}",
            f"<b>{APP_NAME}</b> {__version__} — AI 赛璐璐（平涂）图去噪<br><br>"
            "把纯色块上的颗粒与云雾斑块洗掉，同时保住腮红/阴影这类真实渐变、1–2 像素细笔画与抗锯齿边缘，"
            "并顺手收拾 alpha 通道。<br><br>"
            "命令行版本：<code>celclean clean 图.png</code><br>"
            "「噪点放大」只影响预览，不写进保存的图片。<br><br>"
            f"{describe_settings(self.portable)}",
        )


def selftest(image: Path, out_dir: Path, show: bool, out_json: Path | None, portable: bool = False) -> int:
    """Load, clean and save one image through the real widgets; report what happened as JSON."""
    result: dict = {"ok": False}
    app = QApplication.instance() or QApplication(sys.argv[:1])
    app.setApplicationName(APP_NAME)
    window = MainWindow(portable=portable)
    window.auto_preview.setChecked(False)
    window.resize(1480, 940)
    window.show()
    app.processEvents()
    try:
        started = time.perf_counter()
        window.open_image(image)
        window.run_clean()
        deadline = time.perf_counter() + 900
        while window.out is None and time.perf_counter() < deadline:
            app.processEvents(QEventLoop.ProcessEventsFlag.AllEvents, 50)
            time.sleep(0.005)
        elapsed = time.perf_counter() - started
        app.processEvents()
        shot = out_dir / "window.png"
        window.grab().save(str(shot))
        # a second grab at 1:1: at fit zoom the noise is averaged away by the downscale
        window.view.zoom_reset()
        app.processEvents()
        panes = out_dir / "panes-1to1.png"
        window.view.grab().save(str(panes))
        window.detail_box.setChecked(True)
        app.processEvents()
        detail = out_dir / "panes-detail.png"
        window.view.grab().save(str(detail))
        target = out_dir / f"{image.stem}-clean.png"
        report = target.with_name(target.stem + "-report.json") if window.panel.params().write_report else None
        if window.out is not None:
            window._save_to(target)
        if report is not None:  # written by the worker thread, so wait for the file
            deadline = time.perf_counter() + 300
            while not report.exists() and time.perf_counter() < deadline:
                app.processEvents()
                time.sleep(0.005)
        app.processEvents()
        changed = None
        if window.out is not None and window.rgba is not None:
            changed = float(
                np.abs(window.out[..., :3].astype(np.int16) - window.rgba[..., :3].astype(np.int16)).mean()
            )
        result = {
            "ok": window.out is not None and shot.exists() and target.exists(),
            "platform": app.platformName(),
            "image": str(image),
            "size": list(window.rgba.shape[:2]) if window.rgba is not None else None,
            "window_px": [window.width(), window.height()],
            "screenshot": str(shot),
            "screenshot_bytes": shot.stat().st_size if shot.exists() else 0,
            "output": str(target),
            "output_bytes": target.stat().st_size if target.exists() else 0,
            "report": str(report) if report else None,
            "report_written": bool(report and report.exists()),
            "seconds": round(elapsed, 3),
            "info": window.info,
            "mean_abs_delta_all_px_levels": changed,
            "panes_have_pixmaps": [
                window.view.original.first_image() is not None,
                window.view.cleaned.first_image() is not None,
            ],
            "status": window.state_label.text(),
            "portable_flag": bool(window.portable),  # --portable was passed
            "portable_active": settings_file(portable) is not None,
            "settings": describe_settings(portable),
            "settings_file": str(settings_file(portable)),
            "boost_gain": window.view.gain(),
            "panes_1to1": str(panes),
            "panes_detail": str(detail),
        }
        window._save_settings()  # prove where the settings land, not just that the app ran
    finally:
        window.close()
    text = json.dumps(result, ensure_ascii=False, indent=2, default=str)
    print(text)
    if out_json is not None:
        out_json.write_text(text, encoding="utf-8")
    return 0 if result.get("ok") else 1


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="celclean-gui", description="celclean 桌面版（PySide6）")
    parser.add_argument("file", nargs="?", help="要打开的图片")
    parser.add_argument("--selftest", action="store_true", help="跑一遍自检后退出（打包验证用）")
    parser.add_argument("--selftest-out", default=None, help="自检结果 JSON 写到这个路径")
    parser.add_argument("--show", action="store_true", help="自检时用真实窗口（默认 offscreen，不弹窗）")
    parser.add_argument("--portable", action="store_true",
                        help="便携（绿色）模式：配置文件放 exe 旁边，不写注册表。打包版在可写目录下默认就是它")
    parser.add_argument("--out-dir", default=None, help="自检产物的目录（默认临时目录）")
    parser.add_argument("--version", action="version", version=f"celclean-gui {__version__}")
    args = parser.parse_args(list(sys.argv[1:] if argv is None else argv))

    if args.selftest and not args.show:
        os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
    if sys.platform == "win32":
        try:  # keep the taskbar icon ours instead of python's
            import ctypes

            ctypes.windll.shell32.SetCurrentProcessExplicitAppUserModelID("celclean.gui.1")
        except Exception:
            pass

    app = QApplication.instance() or QApplication(sys.argv[:1])
    app.setApplicationName(APP_NAME)
    app.setOrganizationName(ORG_NAME)
    app.setApplicationDisplayName(APP_NAME)
    app.setApplicationVersion(__version__)
    icon = icon_path()
    if icon is not None:
        app.setWindowIcon(QIcon(str(icon)))

    if args.selftest:
        import tempfile

        default_candidates = [Path.cwd() / "avator.png", Path(__file__).resolve().parents[3] / "avator.png"]
        image = Path(args.file) if args.file else next((p for p in default_candidates if p.exists()), default_candidates[0])
        if not image.exists():
            print(f"selftest: no such image: {image}", file=sys.stderr)
            return 2
        out_dir = Path(args.out_dir) if args.out_dir else Path(tempfile.mkdtemp(prefix="celclean-selftest-"))
        out_dir.mkdir(parents=True, exist_ok=True)
        return selftest(
            image, out_dir, args.show, Path(args.selftest_out) if args.selftest_out else None, args.portable
        )

    window = MainWindow(portable=args.portable)
    if args.file:
        path = Path(args.file)
        if path.exists():
            window.open_image(path)
        else:
            print(f"celclean-gui: no such file: {path}", file=sys.stderr)
    window.show()
    return app.exec()


if __name__ == "__main__":
    raise SystemExit(main())
