from __future__ import annotations

from typing import TYPE_CHECKING, Any

from PyQt6.QtCore import pyqtSignal as Signal
from PyQt6.QtCore import pyqtSlot as Slot
from PyQt6.QtWidgets import QWidget

from ert.config import ParameterConfig

if TYPE_CHECKING:
    from ert.run_models import RunModel


class ExperimentConfigPanel(QWidget):
    experiment_configuration_changed = Signal()
    parameter_configuration_changed = Signal()

    def __init__(self, run_model: type[RunModel]) -> None:
        super().__init__()
        self.setContentsMargins(10, 10, 10, 10)
        self.__run_model = run_model
        self._parameter_snapshot: list[ParameterConfig] | None = None

    @property
    def active_parameters(self) -> list[ParameterConfig] | None:
        return self._parameter_snapshot

    def get_experiment_type(self) -> type[RunModel]:
        return self.__run_model

    def isConfigurationValid(self) -> bool:
        return True

    def get_experiment_arguments(self) -> Any:
        return {}

    @Slot(QWidget)
    def experimentTypeChanged(self, w: QWidget) -> Any:
        pass
