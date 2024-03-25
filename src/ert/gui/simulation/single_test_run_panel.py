from dataclasses import dataclass

from qtpy.QtWidgets import QFormLayout, QLineEdit

from ert.gui.ertnotifier import ErtNotifier
from ert.gui.ertwidgets.copyablelabel import CopyableLabel
from ert.run_models import SingleTestRun

from .simulation_config_panel import SimulationConfigPanel


@dataclass
class Arguments:
    mode: str
    current_ensemble: str
    experiment_name: str


class SingleTestRunPanel(SimulationConfigPanel):
    def __init__(self, run_path: str, notifier: ErtNotifier):
        SimulationConfigPanel.__init__(self, SingleTestRun)
        self.notifier = notifier
        self.setObjectName("Single_test_run_panel")

        layout = QFormLayout()

        self._name_field = QLineEdit()
        self._name_field.setPlaceholderText("single_test_run")
        self._name_field.setMinimumWidth(250)
        layout.addRow("Experiment name:", self._name_field)

        runpath_label = CopyableLabel(text=run_path)
        layout.addRow("Runpath:", runpath_label)

        self.setLayout(layout)

    def getSimulationArguments(self):
        experiment_name = (
            self._name_field.text()
            if self._name_field.text()
            else self._name_field.placeholderText()
        )

        return Arguments(
            "test_run", self.notifier.current_ensemble_name, experiment_name
        )
