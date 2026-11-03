from __future__ import annotations

from PyQt6.QtCore import Qt, pyqtSignal
from PyQt6.QtWidgets import (
    QCheckBox,
    QHBoxLayout,
    QPushButton,
    QVBoxLayout,
    QWidget,
)


class ForwardModelStepSelector(QWidget):
    selectionChanged = pyqtSignal()

    def __init__(self, step_names: list[str], parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self._checkboxes: list[QCheckBox] = []

        select_all_button = QPushButton("Select all")
        select_all_button.clicked.connect(self.select_all)
        deselect_all_button = QPushButton("Deselect all")
        deselect_all_button.clicked.connect(self.deselect_all)

        button_layout = QHBoxLayout()
        button_layout.addWidget(select_all_button)
        button_layout.addWidget(deselect_all_button)
        button_layout.addStretch(1)

        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.addLayout(button_layout)

        for name in step_names:
            checkbox = QCheckBox(name)
            checkbox.setChecked(True)
            checkbox.stateChanged.connect(lambda _: self.selectionChanged.emit())
            self._checkboxes.append(checkbox)
            layout.addWidget(checkbox)
        layout.addStretch(1)

    def selected_indices(self) -> list[int]:
        return [i for i, cb in enumerate(self._checkboxes) if cb.isChecked()]

    def select_all(self) -> None:
        self._set_all(Qt.CheckState.Checked)

    def deselect_all(self) -> None:
        self._set_all(Qt.CheckState.Unchecked)

    def _set_all(self, state: Qt.CheckState) -> None:
        for checkbox in self._checkboxes:
            checkbox.setCheckState(state)
