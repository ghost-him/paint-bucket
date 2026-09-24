"""Image panes: zoom / pan, a screen-fixed checkerboard, and a noise-revealing detail view.

The "噪点放大" view itself -- what it shows, why it is a level residual and not a contrast stretch,
and the measurements behind that -- lives in :mod:`celclean.gui.detail`.  This module only wires it
to widgets.  The field is cached per image, so moving the gain slider re-renders without filtering
again.
"""

from __future__ import annotations

import numpy as np
from PySide6.QtCore import QPointF, QRectF, QSizeF, Qt, Signal
from PySide6.QtGui import QBrush, QColor, QImage, QPainter, QPixmap
from PySide6.QtWidgets import QSplitter, QWidget

from .detail import (
    DETAIL_TARGET,
    GAIN_MAX,
    GAIN_MIN,
    MIN_ALPHA,
    auto_gain,
    detail_field,
    render_detail,
)

MIN_ZOOM = 0.02
MAX_ZOOM = 32.0
CHECKER = 8

_checker_brush: QBrush | None = None


def to_qimage(rgba: np.ndarray) -> QImage:
    arr = np.ascontiguousarray(rgba)
    h, w = arr.shape[:2]
    return QImage(arr.data, w, h, 4 * w, QImage.Format.Format_RGBA8888).copy()


def _checker() -> QBrush:
    global _checker_brush
    if _checker_brush is None:
        tile = QPixmap(2 * CHECKER, 2 * CHECKER)
        tile.fill(QColor(255, 255, 255))
        p = QPainter(tile)
        p.fillRect(0, 0, CHECKER, CHECKER, QColor(226, 226, 226))
        p.fillRect(CHECKER, CHECKER, CHECKER, CHECKER, QColor(226, 226, 226))
        p.end()
        _checker_brush = QBrush(tile)
    return _checker_brush


class ImageView(QWidget):
    """One zoomable, pannable pane.  Emits :attr:`viewChanged` so a peer can mirror it."""

    viewChanged = Signal(float, QPointF)

    def __init__(self, label: str = "", parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self._label = label
        self._image: QImage | None = None
        self._zoom = 1.0
        self._offset = QPointF(0.0, 0.0)
        self._drag_from: QPointF | None = None
        self._background = "checker"
        self._untouched = True  # fit on first paint / resize until the user zooms
        self.setMinimumSize(180, 180)
        self.setCursor(Qt.CursorShape.OpenHandCursor)
        self.setFocusPolicy(Qt.FocusPolicy.StrongFocus)

    # ---- content -------------------------------------------------------------------------
    def set_image(self, image: QImage | None) -> None:
        self._image = image
        if image is not None and self._untouched:
            self.fit()
        self.update()

    def first_image(self) -> QImage | None:
        return self._image

    def set_background(self, mode: str) -> None:
        self._background = mode
        self.update()

    def set_label(self, text: str) -> None:
        if text != self._label:
            self._label = text
            self.update()

    # ---- view state ----------------------------------------------------------------------
    @property
    def zoom(self) -> float:
        return self._zoom

    def view(self) -> tuple[float, QPointF]:
        return self._zoom, QPointF(self._offset)

    def set_view(self, zoom: float, offset: QPointF) -> None:
        self._zoom = float(np.clip(zoom, MIN_ZOOM, MAX_ZOOM))
        self._offset = QPointF(offset)
        self._untouched = False
        self.update()

    def fit(self) -> None:
        if self._image is None or self._image.isNull():
            return
        avail_w = max(self.width() - 8, 1)
        avail_h = max(self.height() - 8, 1)
        zoom = min(avail_w / self._image.width(), avail_h / self._image.height())
        self._zoom = float(np.clip(zoom, MIN_ZOOM, MAX_ZOOM))
        self._center()
        self._untouched = True
        self.update()
        self.viewChanged.emit(self._zoom, QPointF(self._offset))

    def set_zoom(self, zoom: float, anchor: QPointF | None = None) -> None:
        if self._image is None:
            return
        if anchor is None:
            anchor = QPointF(self.width() / 2.0, self.height() / 2.0)
        new = float(np.clip(zoom, MIN_ZOOM, MAX_ZOOM))
        image_point = (anchor - self._offset) / self._zoom
        self._offset = anchor - image_point * new
        self._zoom = new
        self._untouched = False
        self.update()
        self.viewChanged.emit(self._zoom, QPointF(self._offset))

    def _center(self) -> None:
        if self._image is None:
            return
        self._offset = QPointF(
            (self.width() - self._image.width() * self._zoom) / 2.0,
            (self.height() - self._image.height() * self._zoom) / 2.0,
        )

    def _target(self) -> QRectF:
        assert self._image is not None
        return QRectF(
            self._offset,
            QSizeF(self._image.width() * self._zoom, self._image.height() * self._zoom),
        )

    # ---- painting ------------------------------------------------------------------------
    def paintEvent(self, event) -> None:  # noqa: N802 (Qt naming)
        painter = QPainter(self)
        if self._background == "checker":
            painter.fillRect(self.rect(), _checker())
        else:
            painter.fillRect(self.rect(), QColor(255, 255, 255) if self._background == "white" else QColor(0, 0, 0))
        if self._image is not None and not self._image.isNull():
            # nearest neighbour when magnifying: the point of zooming in is to inspect pixels,
            # not to get a smooth guess of them
            painter.setRenderHint(QPainter.RenderHint.SmoothPixmapTransform, self._zoom < 1.0)
            target = self._target()
            painter.drawImage(target, self._image)
            if self._zoom >= 8.0:
                painter.setPen(QColor(0, 0, 0, 40))
                x = target.left()
                while x < target.right():
                    painter.drawLine(QPointF(x, target.top()), QPointF(x, target.bottom()))
                    x += self._zoom
                y = target.top()
                while y < target.bottom():
                    painter.drawLine(QPointF(target.left(), y), QPointF(target.right(), y))
                    y += self._zoom
            painter.setPen(QColor(0, 0, 0, 150))
            painter.setBrush(QColor(255, 255, 255, 190))
            painter.drawRect(4, 4, 10 + 7 * len(self._label), 18)
            painter.drawText(9, 17, self._label)
            painter.drawText(9, self.height() - 8, f"{self._zoom * 100:.0f}%")
        else:
            painter.setPen(QColor(130, 130, 130))
            painter.drawText(self.rect(), Qt.AlignmentFlag.AlignCenter, "拖入图片，或用「打开图片」")

    # ---- interaction ---------------------------------------------------------------------
    def wheelEvent(self, event) -> None:  # noqa: N802
        delta = event.angleDelta().y()
        if not delta or self._image is None:
            return
        self.set_zoom(self._zoom * (1.25 ** (delta / 120.0)), event.position())
        event.accept()

    def mousePressEvent(self, event) -> None:  # noqa: N802
        if event.button() in (Qt.MouseButton.LeftButton, Qt.MouseButton.MiddleButton):
            self._drag_from = event.position()
            self.setCursor(Qt.CursorShape.ClosedHandCursor)

    def mouseMoveEvent(self, event) -> None:  # noqa: N802
        if self._drag_from is None:
            return
        self._offset += event.position() - self._drag_from
        self._drag_from = event.position()
        self._untouched = False
        self.update()
        self.viewChanged.emit(self._zoom, QPointF(self._offset))

    def mouseReleaseEvent(self, event) -> None:  # noqa: N802
        self._drag_from = None
        self.setCursor(Qt.CursorShape.OpenHandCursor)

    def mouseDoubleClickEvent(self, event) -> None:  # noqa: N802
        self.fit()

    def resizeEvent(self, event) -> None:  # noqa: N802
        super().resizeEvent(event)
        if self._image is not None and self._untouched:
            self.fit()


class CompareView(QSplitter):
    """Original and cleaned side by side, sharing zoom / pan / detail view / background."""

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(Qt.Orientation.Horizontal, parent)
        self.original = ImageView("原图")
        self.cleaned = ImageView("清理后")
        self.addWidget(self.original)
        self.addWidget(self.cleaned)
        self.setChildrenCollapsible(False)
        self._syncing = False
        self._sync = True
        self._gain = 1.0
        self._rgba_original: np.ndarray | None = None
        self._rgba_cleaned: np.ndarray | None = None
        self._field_original: np.ndarray | None = None
        self._field_cleaned: np.ndarray | None = None
        self.original.viewChanged.connect(lambda z, o: self._mirror(self.original, self.cleaned, z, o))
        self.cleaned.viewChanged.connect(lambda z, o: self._mirror(self.cleaned, self.original, z, o))

    # ---- content -------------------------------------------------------------------------
    def set_original(self, rgba: np.ndarray | None) -> None:
        self._rgba_original = rgba
        self._field_original = None
        self._refresh()

    def set_cleaned(self, rgba: np.ndarray | None) -> None:
        self._rgba_cleaned = rgba
        self._field_cleaned = None
        self._refresh()

    def has_original(self) -> bool:
        return self._rgba_original is not None

    def has_cleaned(self) -> bool:
        return self._rgba_cleaned is not None

    def gain(self) -> float:
        return self._gain

    def needs_field(self) -> bool:
        """True when turning the detail view on would have to filter an image (slow on 4K)."""
        if self._gain > 1.0:
            return False
        return (self._rgba_original is not None or self._rgba_cleaned is not None) and (
            self._field_original is None or (self._rgba_cleaned is not None and self._field_cleaned is None)
        )

    def auto_gain(self) -> float:
        """Suggested gain for this pair (measures the original's flat-block texture once)."""
        if self._rgba_original is None:
            return float(DETAIL_TARGET)
        if self._field_original is None:
            self._field_original = detail_field(self._rgba_original)
        return auto_gain(self._field_original, self._rgba_original[..., 3] >= MIN_ALPHA)

    def set_gain(self, gain: float) -> None:
        """gain 1.0 = show the real image, >1 = the noise-revealing detail view."""
        self._gain = float(max(1.0, gain))
        self._refresh()

    def _pane_image(self, rgba: np.ndarray | None, field: np.ndarray | None) -> QImage | None:
        if rgba is None:
            return None
        if self._gain <= 1.0:
            return to_qimage(rgba)
        if field is None:
            field = detail_field(rgba)
            if rgba is self._rgba_original:
                self._field_original = field
            elif rgba is self._rgba_cleaned:
                self._field_cleaned = field
        return to_qimage(render_detail(field, self._gain, rgba[..., 3]))

    def _refresh(self) -> None:
        self.original.set_image(self._pane_image(self._rgba_original, self._field_original))
        self.cleaned.set_image(self._pane_image(self._rgba_cleaned, self._field_cleaned))

    # ---- appearance ----------------------------------------------------------------------
    def set_background(self, mode: str) -> None:
        for pane in (self.original, self.cleaned):
            pane.set_background(mode)

    def set_mode(self, mode: str) -> None:
        """'both' | 'original' | 'cleaned'"""
        self.original.setVisible(mode in ("both", "original"))
        self.cleaned.setVisible(mode in ("both", "cleaned"))
        if mode == "both":
            self.setSizes([500, 500])
        self._syncing = True
        self.original.fit()
        self.cleaned.fit()
        self._syncing = False

    def set_sync(self, on: bool) -> None:
        self._sync = bool(on)
        if self._sync:
            self.original.fit()
            self._mirror(self.original, self.cleaned, *self.original.view())

    def _mirror(self, src: ImageView, dst: ImageView, zoom: float, offset: QPointF) -> None:
        if not self._sync or self._syncing:
            return
        self._syncing = True
        dst.set_view(zoom, offset)
        self._syncing = False

    # ---- view commands -------------------------------------------------------------------
    def fit(self) -> None:
        self._syncing = True
        self.original.fit()
        self.cleaned.fit()
        self._syncing = False

    def zoom_reset(self) -> None:
        self._syncing = True
        self.original.set_zoom(1.0)
        self.cleaned.set_view(*self.original.view())
        self._syncing = False
