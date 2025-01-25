from PyQt6.QtCore import QRect, QSize
from PyQt6.QtCore import pyqtSignal as Signal
from PyQt6.QtCore import pyqtSlot as Slot
from PyQt6.QtGui import QColor, QMouseEvent, QPainter, QPaintEvent
from PyQt6.QtWidgets import QColorDialog, QFrame


class ColorBox(QFrame):
    colorChanged = Signal(QColor)
    mouseRelease = Signal()

    """A widget that shows a colored box"""

    def __init__(self, size: int = 15) -> None:
        QFrame.__init__(self)
        self.setFrameStyle(QFrame.Shape.Panel | QFrame.Shadow.Sunken)
        self.setMaximumSize(QSize(size, size))
        self.setMinimumSize(QSize(size, size))

        self._tile_colors = [QColor(255, 255, 255), QColor(200, 200, 255)]
        self._color: QColor = QColor(255, 255, 255)

        self.mouseRelease.connect(self.show_color_dialog)
        self.colorChanged.connect(self.update_color)

    @Slot(QColor)
    def update_color(self, color: QColor) -> None:
        self._color = color
        self.update()

    @Slot()
    def show_color_dialog(self) -> None:
        color_dialog = QColorDialog(self._color, self)
        color_dialog.setWindowTitle("Select color")
        color_dialog.setOption(QColorDialog.ColorDialogOption.ShowAlphaChannel)
        color_dialog.accepted.connect(
            lambda: self.colorChanged.emit(color_dialog.selectedColor())
        )
        color_dialog.open()

    @property
    def color(self) -> QColor:
        return self._color

    @color.setter
    def color(self, color: tuple[str, float]) -> None:
        new_color = QColor(color[0])
        new_color.setAlphaF(color[1])
        self._color = new_color
        self.update()

    def paintEvent(self, a0: QPaintEvent | None) -> None:
        """Paints the box"""
        painter = QPainter(self)
        rect = self.contentsRect()
        tile_count = 3
        tile_size = int(rect.width() / tile_count)
        painter.save()
        painter.translate(rect.x(), rect.y())

        for y in range(tile_count):
            for x in range(tile_count):
                color_index = (y * tile_count + x) % 2
                tile_rect = QRect(x * tile_size, y * tile_size, tile_size, tile_size)
                painter.fillRect(tile_rect, self._tile_colors[color_index])

        painter.restore()
        painter.fillRect(rect, self._color)

        QFrame.paintEvent(self, a0)

    def mouseReleaseEvent(self, a0: QMouseEvent | None) -> None:
        if a0:
            self.mouseRelease.emit()
        return super().mouseReleaseEvent(a0)
