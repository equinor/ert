from dataclasses import dataclass

from qtpy.QtWidgets import QFormLayout

from ert.gui.ertwidgets.caseselector import CaseSelector
from ert.gui.ertwidgets.copyablelabel import CopyableLabel
from ert.gui.ertwidgets.models.activerealizationsmodel import ActiveRealizationsModel
from ert.libres_facade import LibresFacade
from ert.run_models import SingleTestRun

from .simulation_config_panel import SimulationConfigPanel


@dataclass
class Arguments:
    mode: str
    current_case: str


class SingleTestRunPanel(SimulationConfigPanel):
    def __init__(self, ert, notifier):
        self.ert = ert
        self.notifier = notifier
        facade = LibresFacade(ert)
        SimulationConfigPanel.__init__(self, SingleTestRun)
        self.setObjectName("Single_test_run_panel")
        layout = QFormLayout()

        case_selector = CaseSelector(notifier)
        layout.addRow("Current case:", case_selector)

        runpath_label = CopyableLabel(text=facade.run_path)
        layout.addRow("Runpath:", runpath_label)

        self._active_realizations_model = ActiveRealizationsModel(facade)

        self.setLayout(layout)

    def getSimulationArguments(self):
        return Arguments("test_run", self.notifier.current_case_name)
