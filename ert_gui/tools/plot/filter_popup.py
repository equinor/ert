from qtpy.QtCore import Qt, Signal
from qtpy.QtWidgets import (
    QWidget,
    QFrame,
    QDialog,
    QVBoxLayout,
    QCheckBox,
    QLabel,
    QLayout,
)
from qtpy.QtGui import QCursor


class FilterPopup(QDialog):
    filterSettingsChanged = Signal(dict)

    def __init__(self, parent, key_defs):
        QDialog.__init__(
            self,
            parent,
            Qt.WindowStaysOnTopHint
            | Qt.X11BypassWindowManagerHint
            | Qt.FramelessWindowHint,
        )
        self.setVisible(False)

        self.filter_items = {}

        layout = QVBoxLayout()
        layout.setContentsMargins(0, 0, 0, 0)
        frame = QFrame()
        frame.setFrameStyle(QFrame.StyledPanel | QFrame.Raised)
        layout.addWidget(frame)

        self.__layout = QVBoxLayout()
        self.__layout.setSizeConstraint(QLayout.SetFixedSize)
        self.__layout.addWidget(QLabel("Filter by data type:"))

        filters = {k["metadata"]["data_origin"] for k in key_defs}
        for f in filters:
            self.addFilterItem(f, f)

        frame.setLayout(self.__layout)

        self.setLayout(layout)
        self.adjustSize()

    def addFilterItem(self, name, id, value=True):
        self.filter_items[id] = value

        check_box = QCheckBox(name)
        check_box.setChecked(value)

        def toggleItem(checked):
            self.filter_items[id] = checked
            self.filterSettingsChanged.emit(self.filter_items)

        check_box.toggled.connect(toggleItem)

        self.__layout.addWidget(check_box)

    def leaveEvent(self, QEvent):
        QWidget.leaveEvent(self, QEvent)
        self.hide()

    def show(self):
        QWidget.show(self)
        p = QCursor().pos()
        self.move(p.x(), p.y())
