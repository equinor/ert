from __future__ import annotations

from typing import TYPE_CHECKING

from PyQt6.QtCore import QEvent, Qt
from PyQt6.QtCore import pyqtSignal as Signal
from PyQt6.QtGui import QCursor
from PyQt6.QtWidgets import (
    QCheckBox,
    QDialog,
    QFrame,
    QLabel,
    QLayout,
    QVBoxLayout,
    QWidget,
)

if TYPE_CHECKING:
    from ert.gui.tools.plot.plot_api import PlotApiKeyDefinition


class FilterPopup(QDialog):
    filterSettingsChanged = Signal(dict)

    def __init__(
        self, parent: QWidget | None, key_defs: list[PlotApiKeyDefinition]
    ) -> None:
        QDialog.__init__(
            self,
            parent,
            Qt.WindowType.WindowStaysOnTopHint
            | Qt.WindowType.X11BypassWindowManagerHint
            | Qt.WindowType.FramelessWindowHint,
        )
        self.setVisible(False)

        self.filter_items: dict[str, bool] = {}

        layout = QVBoxLayout()
        layout.setContentsMargins(0, 0, 0, 0)
        frame = QFrame()
        frame.setFrameStyle(QFrame.Shape.StyledPanel | QFrame.Shadow.Raised)
        layout.addWidget(frame)

        self.__layout = QVBoxLayout()
        self.__layout.setSizeConstraint(QLayout.SizeConstraint.SetFixedSize)
        self.__layout.addWidget(QLabel("Filter by datatype:"))

        filters = {k.metadata["data_origin"] for k in key_defs}
        for f in filters:
            self.addFilterItem(f, f)

        frame.setLayout(self.__layout)

        self.setLayout(layout)
        self.adjustSize()

    def addFilterItem(self, name: str, id_: str, value: bool = True) -> None:
        self.filter_items[id_] = value

        check_box = QCheckBox(name)
        check_box.setChecked(value)
        check_box.setObjectName("FilterCheckBox")

        def toggleItem(checked: bool) -> None:
            self.filter_items[id_] = checked
            self.filterSettingsChanged.emit(self.filter_items)

        check_box.toggled.connect(toggleItem)

        self.__layout.addWidget(check_box)

    def leaveEvent(self, a0: QEvent | None) -> None:
        self.hide()
        QWidget.leaveEvent(self, a0)

    def show(self) -> None:
        QWidget.show(self)
        p = QCursor().pos()
        self.move(p.x(), p.y())
