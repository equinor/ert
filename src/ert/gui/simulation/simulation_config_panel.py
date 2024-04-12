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

    def _realizations_from_fs(self):
        ensemble = str(self._ensemble_selector.currentText())
        if ensemble:
            mask = self.notifier.storage.get_ensemble_by_name(
                ensemble
            ).get_realization_mask_with_parameters()
            self._active_realizations_field.model.setValueFromMask(mask)
