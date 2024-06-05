from __future__ import annotations

from typing import TYPE_CHECKING, Any, Dict

from qtpy.QtCore import Signal
from qtpy.QtWidgets import QWidget

if TYPE_CHECKING:
    from ert.run_models import BaseRunModel


class ExperimentConfigPanel(QWidget):
    simulationConfigurationChanged = Signal()

    def __init__(self, simulation_model: BaseRunModel) -> None:
        QWidget.__init__(self)
        self.setContentsMargins(10, 10, 10, 10)
        self.__simulation_model = simulation_model

    def get_experiment_type(self) -> BaseRunModel:
        return self.__simulation_model

    @staticmethod
    def isConfigurationValid() -> bool:
        return True

    @staticmethod
    def get_experiment_arguments() -> Dict[str, Any]:
        return {}
