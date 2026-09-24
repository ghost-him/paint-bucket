"""Batch dialog: run the same parameters over a folder (or over what was dropped on the window)."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from PySide6.QtCore import Qt
from PySide6.QtWidgets import (
    QCheckBox,
    QDialog,
    QDialogButtonBox,
    QFileDialog,
    QGridLayout,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QPushButton,
    QVBoxLayout,
    QWidget,
)

from .jobs import IMAGE_SUFFIXES
from .settings import GuiParams


@dataclass
class BatchRequest:
    paths: list[Path]
    out_dir: Path | None  # None = next to each source file
    overwrite: bool
    report: bool


def collect_images(folder: Path, recursive: bool) -> list[Path]:
    pattern = "**/*" if recursive else "*"
    return sorted(p for p in folder.glob(pattern) if p.is_file() and p.suffix.lower() in IMAGE_SUFFIXES)


class BatchDialog(QDialog):
    """Small form; the actual work is started by the main window."""

    def __init__(
        self,
        parent: QWidget | None = None,
        params: GuiParams | None = None,
        paths: list[Path] | None = None,
        folder: Path | None = None,
    ) -> None:
        super().__init__(parent)
        self.setWindowTitle("批处理")
        self.setMinimumWidth(520)
        self._dropped = list(paths or [])
        self._folder = folder

        grid = QGridLayout()
        row = 0

        self.folder_edit = QLineEdit(str(folder) if folder else "")
        self.folder_edit.setPlaceholderText("选择要处理的文件夹…")
        browse = QPushButton("浏览…")
        browse.clicked.connect(self._pick_folder)
        grid.addWidget(QLabel("输入文件夹"), row, 0)
        grid.addWidget(self.folder_edit, row, 1)
        grid.addWidget(browse, row, 2)
        row += 1

        self.recursive = QCheckBox("包含子文件夹")
        grid.addWidget(self.recursive, row, 1)
        row += 1

        self.same_dir = QCheckBox("输出到原目录（文件名加后缀）")
        self.same_dir.setChecked(True)
        self.same_dir.toggled.connect(self._sync_enabled)
        grid.addWidget(self.same_dir, row, 1)
        row += 1

        self.out_edit = QLineEdit()
        self.out_edit.setPlaceholderText("输出文件夹…")
        out_browse = QPushButton("浏览…")
        out_browse.clicked.connect(self._pick_out)
        self._out_browse = out_browse
        grid.addWidget(QLabel("输出文件夹"), row, 0)
        grid.addWidget(self.out_edit, row, 1)
        grid.addWidget(out_browse, row, 2)
        row += 1

        self.overwrite = QCheckBox("覆盖已存在的输出（否则跳过）")
        self.overwrite.setChecked(True)
        grid.addWidget(self.overwrite, row, 1)
        row += 1

        self.report = QCheckBox("同时输出 JSON 报告")
        self.report.setChecked(bool(params.write_report) if params else False)
        grid.addWidget(self.report, row, 1)
        row += 1

        self.info = QLabel()
        self.info.setWordWrap(True)
        grid.addWidget(self.info, row, 1, 1, 2)
        row += 1

        buttons = QDialogButtonBox()
        self.start = buttons.addButton("开始清理", QDialogButtonBox.ButtonRole.AcceptRole)
        buttons.addButton("取消", QDialogButtonBox.ButtonRole.RejectRole)
        buttons.accepted.connect(self.accept)
        buttons.rejected.connect(self.reject)

        layout = QVBoxLayout(self)
        layout.addLayout(grid)
        hint = QLabel("参数取自主窗口的「参数」面板；大图请先把采样步长调成 2。")
        hint.setStyleSheet("color: gray;")
        layout.addWidget(hint)
        layout.addWidget(buttons)

        for widget in (self.folder_edit, self.out_edit):
            widget.textChanged.connect(self._update_info)
        self.recursive.toggled.connect(self._update_info)
        self.same_dir.toggled.connect(self._update_info)
        self._sync_enabled()
        self._update_info()

    # ---- helpers -------------------------------------------------------------------------
    def _sync_enabled(self) -> None:
        custom = not self.same_dir.isChecked()
        self.out_edit.setEnabled(custom)
        self._out_browse.setEnabled(custom)

    def _pick_folder(self) -> None:
        chosen = QFileDialog.getExistingDirectory(self, "选择输入文件夹", self.folder_edit.text())
        if chosen:
            self.folder_edit.setText(chosen)

    def _pick_out(self) -> None:
        chosen = QFileDialog.getExistingDirectory(self, "选择输出文件夹", self.out_edit.text())
        if chosen:
            self.out_edit.setText(chosen)

    def candidate_paths(self) -> list[Path]:
        if self._dropped:
            return self._dropped
        folder = Path(self.folder_edit.text().strip()) if self.folder_edit.text().strip() else self._folder
        if folder is None or not Path(folder).is_dir():
            return []
        return collect_images(Path(folder), self.recursive.isChecked())

    def _update_info(self) -> None:
        count = len(self.candidate_paths())
        self.info.setText(f"将要处理 {count} 个图片文件。" if count else "还没有选择输入。")
        self.start.setEnabled(count > 0)

    def request(self) -> BatchRequest:
        paths = self.candidate_paths()
        out_dir = None
        if not self.same_dir.isChecked() and self.out_edit.text().strip():
            out_dir = Path(self.out_edit.text().strip())
        return BatchRequest(paths=paths, out_dir=out_dir, overwrite=self.overwrite.isChecked(), report=self.report.isChecked())
