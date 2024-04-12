from dataclasses import dataclass
from datetime import datetime

from qtpy.QtWidgets import QFormLayout

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

        runpath_label = CopyableLabel(text=run_path)
        layout.addRow("Runpath:", runpath_label)

        self.setLayout(layout)

    def getSimulationArguments(self):
        ensemble_name = f"{datetime.now().strftime('%Y-%m-%dT%H%M')}"
        return Arguments("test_run", ensemble_name, "single_test_run")
