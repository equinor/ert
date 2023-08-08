from __future__ import annotations

from typing import TYPE_CHECKING, Iterable, Optional

from qtpy.QtCore import Qt, Signal
from qtpy.QtWidgets import QComboBox

from ert.gui.ertnotifier import ErtNotifier

if TYPE_CHECKING:
    from ert.storage import EnsembleReader


class CaseSelector(QComboBox):
    case_populated = Signal()

    def __init__(
        self,
        notifier: ErtNotifier,
        update_ert: bool = True,
        show_only_initialized: bool = False,
        ignore_current: bool = False,
    ):
        super().__init__()
        self.notifier = notifier

        # If true current case of ert will be change
        self._update_ert = update_ert
        # only show initialized cases
        self._show_only_initialized = show_only_initialized
        # ignore the currently selected case if it changes
        self._ignore_current = ignore_current

        self.setSizeAdjustPolicy(QComboBox.AdjustToContents)

        self.setEnabled(False)

        if update_ert:
            # Update ERT when this combo box is changed
            self.currentIndexChanged[int].connect(self._on_current_index_changed)

            # Update this combo box when ERT is changed
            notifier.current_case_changed.connect(self._on_global_current_case_changed)

        notifier.ertChanged.connect(self.populate)
        notifier.storage_changed.connect(self.populate)

        if notifier._storage is not None:
            self.populate()

    def populate(self):
        block = self.blockSignals(True)

        self.clear()

        if self._case_list():
            self.setEnabled(True)

        for case in self._case_list():
            self.addItem(case.name, userData=case)

        if not self._ignore_current:
            current_index = self.findData(
                self.notifier.current_case, Qt.ItemDataRole.UserRole
            )
            self.setCurrentIndex(max(current_index, 0))

        self.blockSignals(block)

        self.case_populated.emit()

    def _case_list(self) -> Iterable[EnsembleReader]:
        if self._show_only_initialized:
            case_list = (x for x in self.notifier.storage.ensembles if x.is_initalized)
        else:
            case_list = self.notifier.storage.ensembles
        return sorted(case_list, key=lambda x: x.started_at, reverse=True)

    def _on_current_index_changed(self, index: int) -> None:
        self.notifier.set_current_case(self.itemData(index))

    def _on_global_current_case_changed(self, data: Optional[EnsembleReader]) -> None:
        self.setCurrentIndex(max(self.findData(data, Qt.ItemDataRole.UserRole), 0))
