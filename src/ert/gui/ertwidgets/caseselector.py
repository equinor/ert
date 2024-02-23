from __future__ import annotations

from typing import TYPE_CHECKING, Optional

from qtpy.QtCore import Qt, Signal
from qtpy.QtWidgets import QComboBox

from ert.gui.presenter import Presenter

if TYPE_CHECKING:
    from ert.storage import EnsembleReader


class CaseSelector(QComboBox):
    case_populated = Signal()

    def __init__(
        self,
        presenter: Presenter,
        update_ert: bool = True,
        show_only_initialized: bool = False,
    ):
        super().__init__()

        # If true current case of ert will be change
        self._update_ert = update_ert
        # only show initialized cases
        self._show_only_initialized = show_only_initialized

        self.setSizeAdjustPolicy(QComboBox.AdjustToContents)

        self.setEnabled(False)

        if update_ert:
            # Update ERT when this combo box is changed
            self.currentIndexChanged[int].connect(self._on_current_index_changed)

            # Update this combo box when ERT is changed
            presenter.current_case_changed.connect(self._on_global_current_case_changed)

        presenter.ert_changed.connect(self.populate)
        presenter.storage_changed.connect(self.populate)

        if self.presenter.current_case is not None:
            self.populate()

    def populate(self):
        block = self.blockSignals(True)

        self.clear()

        case_list = sorted(
            self.presenter.ensembled(initialized=self._show_only_initialized),
            key=lambda x: x.started_at,
            reverse=True,
        )

        if case_list:
            self.setEnabled(True)

        for case in case_list:
            self.addItem(case.name, userData=case)

        current_index = self.findData(
            self.presenter.current_case, Qt.ItemDataRole.UserRole
        )

        self.setCurrentIndex(max(current_index, 0))

        self.blockSignals(block)

        self.case_populated.emit()

    def _on_current_index_changed(self, index: int) -> None:
        self.presenter.current_case = self.itemData(index)

    def _on_global_current_case_changed(self, data: Optional[EnsembleReader]) -> None:
        self.setCurrentIndex(max(self.findData(data, Qt.ItemDataRole.UserRole), 0))
