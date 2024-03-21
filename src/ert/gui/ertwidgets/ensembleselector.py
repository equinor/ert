from __future__ import annotations

from typing import TYPE_CHECKING, Iterable, Optional

from qtpy.QtCore import Qt, Signal
from qtpy.QtWidgets import QComboBox

from ert.gui.ertnotifier import ErtNotifier
from ert.storage.realization_storage_state import RealizationStorageState

if TYPE_CHECKING:
    from ert.storage import Ensemble


class EnsembleSelector(QComboBox):
    ensemble_populated = Signal()

    def __init__(
        self,
        notifier: ErtNotifier,
        update_ert: bool = True,
        show_only_initialized: bool = False,
        show_only_undefined: bool = False,
    ):
        super().__init__()
        self.notifier = notifier

        # If true current ensemble of ert will be change
        self._update_ert = update_ert
        # only show initialized ensembles
        self._show_only_initialized = show_only_initialized
        self._show_only_undefined = show_only_undefined

        self.setSizeAdjustPolicy(QComboBox.AdjustToContents)

        self.setEnabled(False)

        if update_ert:
            # Update ERT when this combo box is changed
            self.currentIndexChanged[int].connect(self._on_current_index_changed)

            # Update this combo box when ERT is changed
            notifier.current_ensemble_changed.connect(
                self._on_global_current_ensemble_changed
            )

        notifier.ertChanged.connect(self.populate)
        notifier.storage_changed.connect(self.populate)

        if notifier.is_storage_available:
            self.populate()

    def populate(self) -> None:
        block = self.blockSignals(True)

        self.clear()

        if self._ensemble_list():
            self.setEnabled(True)

        for ensemble in self._ensemble_list():
            self.addItem(ensemble.name, userData=ensemble)

        current_index = self.findData(
            self.notifier.current_ensemble, Qt.ItemDataRole.UserRole
        )

        self.setCurrentIndex(max(current_index, 0))

        self.blockSignals(block)

        self.ensemble_populated.emit()

    def _ensemble_list(self) -> Iterable[Ensemble]:
        if self._show_only_initialized:
            ensemble_list = (
                x
                for x in self.notifier.storage.ensembles
                if x.is_initalized and not x.has_data()
            )
        elif self._show_only_undefined:
            ensemble_list = (
                ensemble
                for ensemble in self.notifier.storage.ensembles
                if all(
                    e == RealizationStorageState.UNDEFINED
                    for e in ensemble.get_ensemble_state()
                )
            )
        else:
            ensemble_list = self.notifier.storage.ensembles
        return sorted(ensemble_list, key=lambda x: x.started_at, reverse=True)

    def _on_current_index_changed(self, index: int) -> None:
        self.notifier.set_current_ensemble(self.itemData(index))

    def _on_global_current_ensemble_changed(self, data: Optional[Ensemble]) -> None:
        self.setCurrentIndex(max(self.findData(data, Qt.ItemDataRole.UserRole), 0))
