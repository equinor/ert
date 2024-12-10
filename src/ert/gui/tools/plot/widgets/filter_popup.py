from __future__ import annotations

from typing import TYPE_CHECKING

from qtpy.QtCore import QEvent, Qt, Signal
from qtpy.QtGui import QCursor
from qtpy.QtWidgets import (
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
            Qt.WindowStaysOnTopHint  # type: ignore
            | Qt.X11BypassWindowManagerHint  # type: ignore
            | Qt.FramelessWindowHint,  # type: ignore
        )
        self.setVisible(False)

        self.filter_items: dict[str, bool] = {}

        layout = QVBoxLayout()
        layout.setContentsMargins(0, 0, 0, 0)
        frame = QFrame()
        frame.setFrameStyle(QFrame.StyledPanel | QFrame.Raised)
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

    def leaveEvent(self, event: QEvent | None) -> None:
        self.hide()
        QWidget.leaveEvent(self, event)

    def show(self) -> None:
        QWidget.show(self)
        p = QCursor().pos()
        self.move(p.x(), p.y())
