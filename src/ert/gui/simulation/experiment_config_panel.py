from typing import Any, Dict

from qtpy.QtCore import Signal
from qtpy.QtWidgets import QWidget


class ExperimentConfigPanel(QWidget):
    simulationConfigurationChanged = Signal()

    def __init__(self, simulation_model):
        QWidget.__init__(self)
        self.setContentsMargins(10, 10, 10, 10)
        self.__simulation_model = simulation_model

    def get_experiment_type(self):
        return self.__simulation_model

    @staticmethod
    def isConfigurationValid():
        return True

    @staticmethod
    def get_experiment_arguments() -> Dict[str, Any]:
        return {}
