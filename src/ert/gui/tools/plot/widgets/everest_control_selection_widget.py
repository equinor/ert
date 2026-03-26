from __future__ import annotations

from PyQt6.QtCore import Qt, pyqtSignal
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
        self._pinned_control: str | None = None
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

    def set_pinned_control(self, control: str | None) -> None:
        if self._pinned_control == control:
            return
        self._pinned_control = control
        self._controls_list.blockSignals(True)
        self._controls_list.clearSelection()
        matches = self._controls_list.findItems(control, Qt.MatchFlag.MatchExactly)
        for item in matches:
            item.setSelected(True)
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
        if self._pinned_control is not None:
            for i in range(self._controls_list.count()):
                item = self._controls_list.item(i)
                assert item is not None
                if item.text() == self._pinned_control and not item.isSelected():
                    self._controls_list.blockSignals(True)
                    item.setSelected(True)
                    self._controls_list.blockSignals(False)
                    break
        self.controlSelectionChanged.emit()
