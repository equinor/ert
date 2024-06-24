from typing import List, Optional

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

from ert.gui.tools.plot.plot_api import PlotApiKeyDefinition


class FilterPopup(QDialog):
    filterSettingsChanged = Signal(dict)

    def __init__(
        self, parent: Optional[QWidget], key_defs: List[PlotApiKeyDefinition]
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

    def addFilterItem(self, name: str, _id: str, value: bool = True) -> None:
        self.filter_items[_id] = value

        check_box = QCheckBox(name)
        check_box.setChecked(value)
        check_box.setObjectName("FilterCheckBox")

        def toggleItem(checked: bool) -> None:
            self.filter_items[_id] = checked
            self.filterSettingsChanged.emit(self.filter_items)

        check_box.toggled.connect(toggleItem)

        self.__layout.addWidget(check_box)

    def leaveEvent(self, a0: Optional[QEvent]) -> None:
        QWidget.leaveEvent(self, QEvent)  # type: ignore
        self.hide()

    def show(self) -> None:
        QWidget.show(self)
        p = QCursor().pos()
        self.move(p.x(), p.y())
