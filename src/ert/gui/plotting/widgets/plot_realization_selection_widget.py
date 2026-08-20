from PyQt6.QtCore import QSignalBlocker
from PyQt6.QtCore import pyqtSignal as Signal
from PyQt6.QtWidgets import (
    QAbstractItemView,
    QListWidget,
    QListWidgetItem,
    QVBoxLayout,
    QWidget,
)


class RealizationSelectionWidget(QWidget):
    realizationSelectionChanged = Signal()

    def __init__(
        self,
        realizations: list[str],
    ) -> None:
        super().__init__()
        self._realizations_list = QListWidget()
        self._realizations_list.setSelectionMode(
            QAbstractItemView.SelectionMode.SingleSelection
        )
        self._realizations_list.itemSelectionChanged.connect(self._onSelectionChanged)

        layout = QVBoxLayout()
        layout.addWidget(self._realizations_list)
        layout.setContentsMargins(0, 0, 0, 0)
        self.setLayout(layout)

        self.set_realizations(realizations)

    def set_realizations(self, realizations: list[str]) -> None:
        previous_selection = self.get_selected_realization()
        with QSignalBlocker(self._realizations_list):
            self._realizations_list.clear()
            for realization in realizations:
                item = QListWidgetItem(realization)
                self._realizations_list.addItem(item)

            if self._realizations_list.count() > 0:
                selected_row = (
                    realizations.index(previous_selection)
                    if previous_selection in realizations
                    else 0
                )
                self._realizations_list.setCurrentRow(selected_row)

    def get_selected_realization(self) -> str | None:
        selected_realization = self._realizations_list.currentItem()
        if selected_realization:
            return selected_realization.text()
        return None

    def _onSelectionChanged(self) -> None:
        self.realizationSelectionChanged.emit()
