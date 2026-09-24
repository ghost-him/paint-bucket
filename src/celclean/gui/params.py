"""The parameter panel: widgets in front of :mod:`celclean.gui.settings`, plus persistence.

The defaults are the shipped CLI defaults, and the tooltips carry the measured reasons
(see TECHNICAL.md) so a user can decide without reading the documentation first.
"""

from __future__ import annotations

from dataclasses import fields

from PySide6.QtCore import QSettings, Signal
from PySide6.QtGui import QColor
from PySide6.QtWidgets import (
    QCheckBox,
    QColorDialog,
    QComboBox,
    QDoubleSpinBox,
    QFormLayout,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QPushButton,
    QSpinBox,
    QVBoxLayout,
    QWidget,
)

from ..pipeline import parse_color
from .settings import PRESETS, GuiParams, estimate_seconds, params_from_mapping


class ParamsPanel(QWidget):
    changed = Signal()

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self._loading = False
        self._bg_good = "#ffffff"
        root = QVBoxLayout(self)
        root.setContentsMargins(8, 8, 8, 8)
        root.setSpacing(8)

        preset_row = QHBoxLayout()
        preset_row.addWidget(QLabel("预设"))
        self.preset = QComboBox()
        self.preset.addItems(["自定义", *PRESETS])
        self.preset.currentTextChanged.connect(self._apply_preset)
        preset_row.addWidget(self.preset, 1)
        root.addLayout(preset_row)

        # --- quality / speed ---------------------------------------------------------------
        core = QGroupBox("质量 / 速度")
        core_form = QFormLayout(core)
        self.stride = QSpinBox()
        self.stride.setRange(1, 4)
        self.stride.setToolTip(
            "窗口采样步长。1 = 每个 tap 都算（默认的质量工作点）；2 快 3.4 倍，但纯色块会退回一档。"
        )
        core_form.addRow("采样步长", self.stride)

        self.auto_radius = QCheckBox("自动（按尺寸推算）")
        self.auto_radius.setToolTip("1254² 时半径为 16，随图片面积开方缩放，上限 24。")
        self.radius = QSpinBox()
        self.radius.setRange(3, 64)
        self.radius.setSuffix(" px")
        self.radius.setToolTip(
            "核心窗口半径。想更干净可试 20；噪声很轻的图反而该调小（8~12），否则会过度平滑。"
        )
        radius_row = QHBoxLayout()
        radius_row.addWidget(self.auto_radius)
        radius_row.addWidget(self.radius, 1)
        core_form.addRow("半径", radius_row)

        self.strength = QDoubleSpinBox()
        self.strength.setRange(0.2, 3.0)
        self.strength.setSingleStep(0.1)
        self.strength.setDecimals(1)
        self.strength.setToolTip(
            "颜色相似度阈值倍数。调大不会更干净，只会把最坏改动从 7 色阶推到 21 色阶；"
            "想更保守（厚涂 / 照片）调小到 0.4~0.6。"
        )
        core_form.addRow("强度", self.strength)
        root.addWidget(core)

        # --- snap --------------------------------------------------------------------------
        snap_box = QGroupBox("纯净色块（snap）")
        snap_form = QFormLayout(snap_box)
        self.snap = QCheckBox("把整块刷成一个颜色")
        self.snap.setToolTip(
            "整块重绘：纯色块最纯（3×3 纯色占比 0.55 → 0.64），但大片柔和渐变上会出现硬边色斑，"
            "所以默认关闭。"
        )
        snap_form.addRow(self.snap)
        self.snap_mode = QComboBox()
        self.snap_mode.addItem("平面（跟随缓坡，推荐）", "plane")
        self.snap_mode.addItem("常数（整块一个颜色，会把缓坡刷平）", "constant")
        snap_form.addRow("块模型", self.snap_mode)
        self.snap_tol = QDoubleSpinBox()
        self.snap_tol.setRange(0.05, 3.0)
        self.snap_tol.setSingleStep(0.05)
        self.snap_tol.setDecimals(2)
        self.snap_tol.setToolTip("离块均值多远（dE）还算平坦；调小保留更多阴影。")
        snap_form.addRow("平坦容差", self.snap_tol)
        self.min_block = QSpinBox()
        self.min_block.setRange(0, 4096)
        self.min_block.setSuffix(" px")
        self.min_block.setToolTip("小于这么多像素的块交给去噪器处理，不做重绘。")
        snap_form.addRow("最小块", self.min_block)
        self.aa_band = QSpinBox()
        self.aa_band.setRange(0, 8)
        self.aa_band.setSuffix(" px")
        self.aa_band.setToolTip("色块边缘保留不重刷的宽度（保护抗锯齿）。只在 snap 打开时生效。")
        snap_form.addRow("边缘保护", self.aa_band)
        self.plane_tol = QDoubleSpinBox()
        self.plane_tol.setRange(1.0, 20.0)
        self.plane_tol.setSingleStep(0.5)
        self.plane_tol.setDecimals(1)
        self.plane_tol.setToolTip(
            "平面模型的放松倍数：残差不超过中位残差的这么多倍才算平面，超出（曲面阴影、漏进来的边缘）"
            "就交给去噪器。"
        )
        snap_form.addRow("平面容差", self.plane_tol)
        self.slope_tol = QDoubleSpinBox()
        self.slope_tol.setRange(0.0, 0.5)
        self.slope_tol.setSingleStep(0.01)
        self.slope_tol.setDecimals(2)
        self.slope_tol.setToolTip("相邻两块坡度差小于这个值（dE/px）时才合并。")
        snap_form.addRow("合并坡度差", self.slope_tol)
        root.addWidget(snap_box)

        # --- alpha -------------------------------------------------------------------------
        alpha_box = QGroupBox("透明通道")
        alpha_form = QFormLayout(alpha_box)
        self.alpha = QComboBox()
        self.alpha.addItem("归一（内部半透明 → 不透明，推荐）", "normalize")
        self.alpha.addItem("保留原样（输出保留透明）", "keep")
        self.alpha.addItem("换成纯色背景（输出不透明）", "flatten")
        self.alpha.setToolTip(
            "AI 出图的内部 alpha 常是 250~254；归一后合成到深色背景上最多亮 5 个色阶（约 2% 像素）。\n"
            "「换成纯色背景」= 归一 + 按 alpha 把整图合成到指定颜色上，输出是不含 alpha 通道的 RGB。"
        )
        alpha_form.addRow(self.alpha)

        self.bg_button = QPushButton()
        self.bg_button.setToolTip(
            "点这里选背景色；也可以直接填 #rrggbb（或 white / black）。\n"
            "合成是逐像素按 alpha 混合，半透明的抗锯齿边缘会正确过渡，不会出现黑边。"
        )
        self.bg_button.clicked.connect(self._pick_background)
        self.bg_hex = QLineEdit()
        self.bg_hex.setFixedWidth(96)
        self.bg_hex.setToolTip("背景色，#rrggbb")
        self.bg_hex.editingFinished.connect(self._background_edited)
        bg_row = QHBoxLayout()
        bg_row.addWidget(self.bg_button, 1)
        bg_row.addWidget(self.bg_hex)
        alpha_form.addRow("背景色", bg_row)
        root.addWidget(alpha_box)

        # --- output ------------------------------------------------------------------------
        out_box = QGroupBox("输出")
        out_form = QFormLayout(out_box)
        self.suffix = QLineEdit()
        self.suffix.setToolTip("输出文件名后缀，结果固定存成 PNG。")
        out_form.addRow("文件名后缀", self.suffix)
        self.write_report = QCheckBox("同时输出 JSON 报告")
        self.write_report.setToolTip("包含验收协议的全部数字（局部起伏、最坏改动、白底渲染口径等）。")
        out_form.addRow(self.write_report)
        root.addWidget(out_box)

        reset = QPushButton("恢复默认参数")
        reset.clicked.connect(self.reset)
        root.addWidget(reset)
        root.addStretch(1)

        for widget in (
            self.stride,
            self.radius,
            self.auto_radius,
            self.strength,
            self.snap,
            self.snap_mode,
            self.snap_tol,
            self.min_block,
            self.aa_band,
            self.plane_tol,
            self.slope_tol,
            self.alpha,
            self.suffix,
            self.write_report,
        ):
            self._connect(widget)
        self._sync_enabled()
        self.set_defaults()

    # ---- signal plumbing -----------------------------------------------------------------
    def _connect(self, widget: QWidget) -> None:
        if isinstance(widget, QCheckBox):
            widget.toggled.connect(self._on_change)
        elif isinstance(widget, QComboBox):
            widget.currentIndexChanged.connect(self._on_change)
        elif isinstance(widget, (QSpinBox, QDoubleSpinBox)):
            widget.valueChanged.connect(self._on_change)
        elif isinstance(widget, QLineEdit):
            widget.textChanged.connect(self._on_change)

    def _on_change(self, *_args) -> None:
        self._sync_enabled()
        if not self._loading:
            self.preset.blockSignals(True)
            self.preset.setCurrentText("自定义")
            self.preset.blockSignals(False)
            self.changed.emit()

    def _pick_background(self) -> None:
        chosen = QColorDialog.getColor(QColor(self._bg_good), self, "选择背景色")
        if chosen.isValid():
            self.bg_hex.setText(chosen.name())
            self._background_edited()

    def _background_edited(self) -> None:
        try:
            color = parse_color(self.bg_hex.text().strip() or "#ffffff")
        except ValueError:
            self.bg_hex.setText(self._bg_good)  # keep the last good value instead of guessing
            return
        self._bg_good = f"#{color[0]:02x}{color[1]:02x}{color[2]:02x}"
        self.bg_hex.setText(self._bg_good)
        self._swatch()
        self._on_change()

    def _swatch(self) -> None:
        color = QColor(self._bg_good)
        text = "#000000" if color.lightness() > 128 else "#ffffff"
        self.bg_button.setText(self._bg_good)
        self.bg_button.setStyleSheet(
            f"background-color: {self._bg_good}; color: {text}; border: 1px solid #888888;"
        )

    def _sync_enabled(self) -> None:
        self.radius.setEnabled(not self.auto_radius.isChecked())
        flatten = self.alpha.currentData() == "flatten"
        self.bg_button.setEnabled(flatten)
        self.bg_hex.setEnabled(flatten)
        for widget in (self.snap_mode, self.snap_tol, self.min_block, self.aa_band, self.plane_tol, self.slope_tol):
            widget.setEnabled(self.snap.isChecked())

    # ---- values --------------------------------------------------------------------------
    def set_defaults(self) -> None:
        self.set_params(GuiParams())

    def reset(self) -> None:
        self.preset.blockSignals(True)
        self.preset.setCurrentText("默认（推荐）")
        self.preset.blockSignals(False)
        self.set_params(GuiParams())
        self.changed.emit()

    def _apply_preset(self, name: str) -> None:
        overrides = PRESETS.get(name)
        if overrides is None:  # "自定义"
            return
        self.set_params(GuiParams(**overrides))
        self.changed.emit()

    def params(self) -> GuiParams:
        return GuiParams(
            stride=self.stride.value(),
            radius=0 if self.auto_radius.isChecked() else self.radius.value(),
            strength=self.strength.value(),
            snap=self.snap.isChecked(),
            snap_mode=self.snap_mode.currentData(),
            snap_tol=self.snap_tol.value(),
            min_block=self.min_block.value(),
            plane_tol=self.plane_tol.value(),
            slope_tol=self.slope_tol.value(),
            aa_band=self.aa_band.value(),
            alpha=self.alpha.currentData(),
            bg_color=self._bg_good,
            sigma=0.0,
            suffix=self.suffix.text().strip() or "-clean",
            write_report=self.write_report.isChecked(),
        )

    def set_params(self, params: GuiParams) -> None:
        self._loading = True
        self.stride.setValue(params.stride)
        self.auto_radius.setChecked(not params.radius)
        self.radius.setValue(params.radius or 16)
        self.strength.setValue(params.strength)
        self.snap.setChecked(params.snap)
        self.snap_mode.setCurrentIndex(max(0, self.snap_mode.findData(params.snap_mode)))
        self.snap_tol.setValue(params.snap_tol)
        self.min_block.setValue(params.min_block)
        self.aa_band.setValue(params.aa_band)
        self.plane_tol.setValue(params.plane_tol)
        self.slope_tol.setValue(params.slope_tol)
        self.alpha.setCurrentIndex(max(0, self.alpha.findData(params.alpha)))
        self._bg_good = params.bg_color
        self.bg_hex.setText(params.bg_color)
        self._swatch()
        self.suffix.setText(params.suffix)
        self.write_report.setChecked(params.write_report)
        self._loading = False
        self._sync_enabled()

    # ---- persistence ---------------------------------------------------------------------
    def save(self, settings: QSettings) -> None:
        settings.beginGroup("params")
        for field in fields(GuiParams):
            settings.setValue(field.name, getattr(self.params(), field.name))
        settings.endGroup()

    def load(self, settings: QSettings) -> None:
        """Restore the panel. No catch-all here: a silent fallback once hid the fact that an INI
        store returns strings and the values never made it back into the widgets."""
        if not settings.contains("params/stride"):
            return
        settings.beginGroup("params")
        raw = {f.name: settings.value(f.name, getattr(GuiParams(), f.name)) for f in fields(GuiParams)}
        settings.endGroup()
        self.set_params(params_from_mapping(raw))
