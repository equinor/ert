from __future__ import annotations

import logging
from collections.abc import Iterable
from typing import TYPE_CHECKING
from uuid import UUID

from PyQt6.QtCore import Qt
from PyQt6.QtCore import pyqtSignal as Signal
from PyQt6.QtWidgets import QComboBox

from ert.gui.ertnotifier import ErtNotifier
from ert.storage.realization_storage_state import RealizationStorageState

from ...config import ErrorInfo
from ..suggestor import Suggestor

if TYPE_CHECKING:
    from ert.storage import Ensemble

logger = logging.getLogger(__name__)


class EnsembleSelector(QComboBox):
    ensemble_populated = Signal()

    def __init__(
        self,
        notifier: ErtNotifier,
        update_ert: bool = True,
        show_only_undefined: bool = False,
        show_only_no_children: bool = False,
    ) -> None:
        super().__init__()
        self.notifier = notifier

        # If true current ensemble of ert will be change
        self._update_ert = update_ert
        # only show initialized ensembles
        self._show_only_undefined = show_only_undefined
        # If True, we filter out any ensembles which have children
        # One use case is if a user wants to rerun because of failures
        # not related to parameterization. We can allow that, but only
        # if the ensemble has not been used in an update, as that would
        # invalidate the result
        self._show_only_no_children = show_only_no_children
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

        try:
            if self._ensemble_list():
                self.setEnabled(True)

            for ensemble in self._ensemble_list():
                self.addItem(
                    f"{ensemble.experiment.name} : {ensemble.name}",
                    userData=str(ensemble.id),
                )
        except OSError as err:
            logger.error(str(err))
            Suggestor(
                errors=[ErrorInfo(str(err))],
                widget_info='<p style="font-size: 28px;">Error writing to storage</p>',
                parent=self,
            ).show()
            return

        current_index = (
            self.findData(
                str(self.notifier.current_ensemble.id), Qt.ItemDataRole.UserRole
            )
            if self.notifier.current_ensemble is not None
            else -1
        )

        self.setCurrentIndex(max(current_index, 0))

        self.blockSignals(block)

        self.ensemble_populated.emit()

    def _ensemble_list(self) -> Iterable[Ensemble]:
        if self._show_only_undefined:
            ensembles = (
                ensemble
                for ensemble in self.notifier.storage.ensembles
                if all(
                    RealizationStorageState.UNDEFINED in e
                    for e in ensemble.get_ensemble_state()
                )
            )
        else:
            ensembles = self.notifier.storage.ensembles
        ensemble_list = list(ensembles)
        if self._show_only_no_children:
            parents = [
                ens.parent for ens in self.notifier.storage.ensembles if ens.parent
            ]
            ensemble_list = [val for val in ensemble_list if val.id not in parents]
        return sorted(ensemble_list, key=lambda x: x.started_at, reverse=True)

    def _on_current_index_changed(self, index: int) -> None:
        self.notifier.set_current_ensemble_id(UUID(self.itemData(index)))

    def _on_global_current_ensemble_changed(self, data: UUID | None) -> None:
        self.setCurrentIndex(max(self.findData(str(data), Qt.ItemDataRole.UserRole), 0))
