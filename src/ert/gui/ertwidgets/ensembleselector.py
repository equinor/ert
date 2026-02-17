from __future__ import annotations

import logging
from collections.abc import Callable, Iterable
from typing import TYPE_CHECKING
from uuid import UUID

from PyQt6.QtCore import Qt
from PyQt6.QtCore import pyqtSignal as Signal
from PyQt6.QtWidgets import QComboBox

from ert.config import ErrorInfo
from ert.gui.ertnotifier import ErtNotifier
from ert.storage import RealizationStorageState

from .suggestor import Suggestor

if TYPE_CHECKING:
    from ert.storage import Ensemble

logger = logging.getLogger(__name__)


class EnsembleSelector(QComboBox):
    """A combo box for selecting an ensemble from the storage.
    Parameters
    ----------
    notifier: ErtNotifier
    update_ert: bool, optional
        If True, changing the selection in this combo box will update the
        current ensemble in ERT.
    filters: list of callables, optional
        A list of "or" filter functions to apply to the ensemble list. If
        provided, only ensembles that pass at least one filter will be shown.
        Default is None, which means no filtering is applied.
    """

    ensemble_populated = Signal()

    def __init__(
        self,
        notifier: ErtNotifier,
        update_ert: bool = True,
        filters: list[Callable[[Iterable[Ensemble]], Iterable[Ensemble]]] | None = None,
    ) -> None:
        super().__init__()
        self.notifier = notifier

        # If true current ensemble of ert will be changed
        self._update_ert = update_ert

        if filters is None:
            self._or_filters = []
        else:
            self._or_filters = filters

        self.setSizeAdjustPolicy(QComboBox.SizeAdjustPolicy.AdjustToContents)

        self.setEnabled(False)

        if update_ert:
            # Update ERT when this combo box is changed
            self.currentIndexChanged.connect(self._on_current_index_changed)

            # Update this combo box when ERT is changed
            notifier.current_ensemble_changed.connect(
                self._on_global_current_ensemble_changed
            )

        notifier.ertChanged.connect(self.populate)

        if notifier.is_storage_available:
            self.populate()

    @property
    def selected_ensemble(self) -> Ensemble | None:
        try:
            return self.notifier.storage.get_ensemble(
                self.itemData(self.currentIndex())
            )
        except KeyError:
            return None

    def populate(self) -> None:
        block = self.blockSignals(True)

        self.clear()
        ensemble_list = list(self._ensemble_list())

        try:
            if ensemble_list:
                self.setEnabled(True)

            for ensemble in ensemble_list:
                self.addItem(
                    f"{ensemble.experiment.name} : {ensemble.name}",
                    userData=str(ensemble.id),
                )
            if ensemble_list:
                first_ensemble_id = str(ensemble_list[0].id)
                current_index = self.findData(
                    first_ensemble_id, Qt.ItemDataRole.UserRole
                )
                self.setCurrentIndex(max(current_index, 0))

        except OSError as err:
            logger.error(str(err))
            Suggestor(
                errors=[ErrorInfo(str(err))],
                widget_info='<p style="font-size: 28px;">Error writing to storage</p>',
                parent=self,
            ).show()
            return

        self.blockSignals(block)
        self.ensemble_populated.emit()

    def _ensemble_list(self) -> Iterable[Ensemble]:
        if not self._or_filters:
            return self.sort_ensembles(self.notifier.storage.ensembles)

        all_ensembles = list(self.notifier.storage.ensembles)
        filtered_ensembles = []

        for filter_func in self._or_filters:
            filtered_ensembles.extend(list(filter_func(all_ensembles)))

        unique_filtered_ensembles = list(dict.fromkeys(filtered_ensembles))
        return self.sort_ensembles(unique_filtered_ensembles)

    @classmethod
    def sort_ensembles(cls, ensemble_list: Iterable[Ensemble]) -> Iterable[Ensemble]:
        return sorted(
            ensemble_list,
            key=lambda e: (
                any(
                    RealizationStorageState.FAILURE_IN_CURRENT in s
                    for s in e.get_ensemble_state()
                ),
                e.started_at,
            ),
            reverse=True,
        )

    def _on_current_index_changed(self, index: int) -> None:
        self.notifier.set_current_ensemble_id(UUID(self.itemData(index)))

    def _on_global_current_ensemble_changed(self, data: UUID | None) -> None:
        self.setCurrentIndex(max(self.findData(str(data), Qt.ItemDataRole.UserRole), 0))
