from __future__ import annotations

import logging
from collections.abc import Callable, Iterable

from PyQt6.QtCore import Qt
from PyQt6.QtCore import pyqtSignal as Signal
from PyQt6.QtWidgets import QComboBox

from ert.config import ErrorInfo
from ert.gui.ertnotifier import ErtNotifier
from ert.gui.utils import truncate_dropdown_item
from ert.storage import Ensemble, RealizationStorageState

from .suggestor import Suggestor

logger = logging.getLogger(__name__)


class EnsembleSelector(QComboBox):
    """A combo box for selecting an ensemble from the storage.
    Parameters
    ----------
    notifier: ErtNotifier
    filters: iterable of callables, optional
        An iterable of "or" filter functions to apply to the ensemble list. If
        provided, only ensembles that pass at least one filter will be shown.
        Default is an empty tuple, which means no filtering is applied.
    """

    ensemble_populated = Signal()
    ensemble_selected = Signal(Ensemble)

    def __init__(
        self,
        notifier: ErtNotifier,
        *,
        filters: Iterable[Callable[[Iterable[Ensemble]], Iterable[Ensemble]]] = (),
    ) -> None:
        super().__init__()

        self.notifier = notifier
        self._or_filters = filters
        self.setSizeAdjustPolicy(QComboBox.SizeAdjustPolicy.AdjustToContents)
        self.setEnabled(False)

        notifier.ertChanged.connect(self.populate)
        self.currentIndexChanged.connect(self._on_current_index_changed)

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

    def _on_current_index_changed(self, _: int) -> None:
        ensemble = self.selected_ensemble
        if ensemble:
            self.ensemble_selected.emit(ensemble)

    def populate(self) -> None:
        block = self.blockSignals(True)

        self.clear()
        ensemble_list: list[Ensemble] = list(self._ensemble_list())

        if ensemble_list:
            self.setEnabled(True)
        try:
            for ensemble in ensemble_list:
                self.addItem(
                    f"{truncate_dropdown_item(ensemble.experiment.name)}"
                    f" : {ensemble.name}",
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
        finally:
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
