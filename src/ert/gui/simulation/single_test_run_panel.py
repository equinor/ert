from dataclasses import dataclass

from qtpy.QtWidgets import QFormLayout

from ert.gui.ertwidgets.caseselector import CaseSelector
from ert.gui.ertwidgets.copyablelabel import CopyableLabel
from ert.gui.ertwidgets.models.activerealizationsmodel import ActiveRealizationsModel
from ert.run_models import SingleTestRun

from .simulation_config_panel import SimulationConfigPanel


@dataclass
class Arguments:
    mode: str
    current_case: str


class SingleTestRunPanel(SimulationConfigPanel):
    def __init__(self, run_path, notifier, ensemble_size: int):
        self.notifier = notifier
        SimulationConfigPanel.__init__(self, SingleTestRun)
        self.setObjectName("Single_test_run_panel")
        layout = QFormLayout()

        case_selector = CaseSelector(notifier)
        layout.addRow("Current case:", case_selector)

        runpath_label = CopyableLabel(text=run_path)
        layout.addRow("Runpath:", runpath_label)

        self._active_realizations_model = ActiveRealizationsModel(ensemble_size)

        self.setLayout(layout)

    def getSimulationArguments(self):
        return Arguments("test_run", self.notifier.current_case_name)
