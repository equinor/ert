import datetime
from typing import Optional, Union

from qtpy.QtCore import QDate
from qtpy.QtGui import QIcon
from qtpy.QtWidgets import (
    QCalendarWidget,
    QHBoxLayout,
    QMenu,
    QToolButton,
    QWidget,
    QWidgetAction,
)

from ert.gui.tools.plot.widgets.clearable_line_edit import ClearableLineEdit


class CustomDateEdit(QWidget):
    def __init__(self) -> None:
        QWidget.__init__(self)
        self._line_edit = ClearableLineEdit()

        self._calendar_button = QToolButton()
        self._calendar_button.setPopupMode(QToolButton.InstantPopup)
        self._calendar_button.setFixedSize(26, 26)
        self._calendar_button.setAutoRaise(True)
        self._calendar_button.setIcon(QIcon("img:calendar_date_range.svg"))
        self._calendar_button.setStyleSheet(
            "QToolButton::menu-indicator { image: none; }"
        )

        tool_menu = QMenu(self._calendar_button)
        self._calendar_widget = QCalendarWidget(tool_menu)
        action = QWidgetAction(tool_menu)
        action.setDefaultWidget(self._calendar_widget)
        tool_menu.addAction(action)
        self._calendar_button.setMenu(tool_menu)

        layout = QHBoxLayout()
        layout.setContentsMargins(0, 0, 0, 0)
        layout.addWidget(self._line_edit)
        layout.addWidget(self._calendar_button)
        self.setLayout(layout)

        self._calendar_widget.activated.connect(self.setDate)

    def setDate(self, date: Union[datetime.date, QDate]) -> None:
        if isinstance(date, datetime.date):
            date = QDate(date.year, date.month, date.day)  # type: ignore

        if date is not None and date.isValid():
            self._line_edit.setText(str(date.toString("yyyy-MM-dd")))
        else:
            self._line_edit.setText("")

    def date(self) -> Optional[datetime.date]:
        date_string = self._line_edit.text()
        if len(str(date_string).strip()) > 0:
            date = QDate.fromString(date_string, "yyyy-MM-dd")
            if date.isValid():
                return datetime.date(date.year(), date.month(), date.day())

        return None
