from typing import Any, Dict

from qtpy.QtCore import Signal
from qtpy.QtWidgets import QWidget


class SimulationConfigPanel(QWidget):
    simulationConfigurationChanged = Signal()

    def __init__(self, simulation_model):
        QWidget.__init__(self)
        self.setContentsMargins(10, 10, 10, 10)
        self.__simulation_model = simulation_model

    def getSimulationModel(self):
        return self.__simulation_model

    @staticmethod
    def isConfigurationValid():
        return True

    @staticmethod
    def getSimulationArguments() -> Dict[str, Any]:
        return {}
