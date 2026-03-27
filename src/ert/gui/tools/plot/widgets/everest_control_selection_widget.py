from __future__ import annotations

from PyQt6.QtCore import pyqtSignal
from PyQt6.QtWidgets import (
    QAbstractItemView,
    QListWidget,
    QListWidgetItem,
    QVBoxLayout,
    QWidget,
)


class EverestControlSelectionWidget(QWidget):
    controlSelectionChanged = pyqtSignal()

    def __init__(self, controls: list[str]) -> None:
        super().__init__()
        self._controls_list = QListWidget()
        self._controls_list.setSelectionMode(
            QAbstractItemView.SelectionMode.MultiSelection
        )
        self._controls_list.itemSelectionChanged.connect(self._onSelectionChanged)

        layout = QVBoxLayout()
        layout.addWidget(self._controls_list)
        layout.setContentsMargins(0, 0, 0, 0)
        self.setLayout(layout)

        self.set_controls(controls)

    def set_controls(self, controls: list[str]) -> None:
        self._controls_list.blockSignals(True)
        self._controls_list.clear()
        for control in controls:
            item = QListWidgetItem(control)
            self._controls_list.addItem(item)
        self._controls_list.blockSignals(False)

    def get_selected_controls(self) -> list[str]:
        selected = []
        for i in range(self._controls_list.count()):
            item = self._controls_list.item(i)
            assert item is not None
            if item.isSelected():
                selected.append(item.text())
        return selected

    def _onSelectionChanged(self) -> None:
        self.controlSelectionChanged.emit()
