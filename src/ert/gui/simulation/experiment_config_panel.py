from __future__ import annotations

from typing import TYPE_CHECKING, Any

from PyQt6.QtCore import pyqtSignal as Signal
from PyQt6.QtCore import pyqtSlot as Slot
from PyQt6.QtWidgets import QWidget

if TYPE_CHECKING:
    from ert.run_models import RunModel


class ExperimentConfigPanel(QWidget):
    simulationConfigurationChanged = Signal()

    def __init__(self, simulation_model: type[RunModel]) -> None:
        super().__init__()
        self.setContentsMargins(10, 10, 10, 10)
        self.__simulation_model = simulation_model

    def get_experiment_type(self) -> type[RunModel]:
        return self.__simulation_model

    def isConfigurationValid(self) -> bool:
        return True

    def get_experiment_arguments(self) -> Any:
        return {}

    @Slot(QWidget)
    def experimentTypeChanged(self, w: QWidget) -> Any:
        pass
