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
        self._realizations_list.clear()
        for realization in realizations:
            item = QListWidgetItem(realization)
            self._realizations_list.addItem(item)

        if self._realizations_list.count() > 0:
            first_item = self._realizations_list.item(0)
            if first_item is not None:
                first_item.setSelected(True)
                self._realizations_list.setCurrentItem(first_item)

    def get_selected_realization(self) -> str | None:
        selected_realization = self._realizations_list.currentItem()
        if selected_realization:
            return selected_realization.text()
        return None

    def _onSelectionChanged(self) -> None:
        self.realizationSelectionChanged.emit()
