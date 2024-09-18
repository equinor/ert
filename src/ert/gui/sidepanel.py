from __future__ import annotations

from typing import List

from qtpy.QtCore import QRectF, Slot
from qtpy.QtGui import QBrush, QColor, QPainter
from qtpy.QtWidgets import (
    QFrame,
    QPushButton,
    QVBoxLayout,
)

from ert.ensemble_evaluator.state import REAL_STATE_TO_COLOR
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
        push_button = ErtStatusButton()
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

        if isinstance(run_dialog, RunDialog):
            run_dialog.progress_update_event.connect(push_button.update_values)

    def select_widget(self) -> None:
        button = self.sender()
        if isinstance(button, ErtStatusButton):
            index = int(button.property("index"))

            for i in range(len(self._widget_list)):
                should_be_visible = i == index
                self._widget_list[i].setVisible(should_be_visible)


class ErtStatusButton(QPushButton):
    def __init__(self):
        super().__init__()
        self.status_dict: dict[str, int] = {}
        self.job_count: int = 1

    def update_values(self, d: dict, count: int) -> None:
        self.status_dict = d
        self.job_count = count
        self.repaint()

    def paintEvent(self, event):
        super().paintEvent(event)
        painter = QPainter(self)

        if self.status_dict and "Finished" in self.status_dict:
            value = self.status_dict["Finished"]
            ratio = value / self.job_count
            painter.setBrush(QBrush(QColor(*REAL_STATE_TO_COLOR["Finished"])))
            painter.drawRect(QRectF(0, 0, ratio * self.width(), self.height()))

        painter.end()
