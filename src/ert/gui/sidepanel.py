from __future__ import annotations

from typing import List

from qtpy.QtCore import Slot
from qtpy.QtWidgets import (
    QFrame,
    QPushButton,
    QVBoxLayout,
)

from ert.gui.simulation.run_dialog import RunDialog


class SidePanel(QFrame):
    def __init__(self) -> None:
        QFrame.__init__(self)

        self.setMinimumWidth(100)
        self.setMinimumHeight(300)

        self._widget_list: List[RunDialog] = []
        self._layout = QVBoxLayout(self)

    @Slot(object)
    def slot_add_widget(self, run_dialog: RunDialog, text: str = "") -> None:
        push_button = QPushButton(self)
        push_button.setMinimumSize(50, 50)
        push_button.setProperty("index", len(self._widget_list))
        push_button.setText(text)

        if not self._widget_list:
            self._layout.addWidget(push_button)
            self._layout.addStretch()
        else:
            self._layout.insertWidget(self._layout.count() - 1, push_button)

        self._widget_list.append(run_dialog)
        push_button.clicked.connect(self.select_widget)

    def select_widget(self) -> None:
        button = self.sender()
        if isinstance(button, QPushButton):
            index = int(button.property("index"))

            for i in range(len(self._widget_list)):
                should_be_visible = i == index
                self._widget_list[i].setVisible(should_be_visible)
