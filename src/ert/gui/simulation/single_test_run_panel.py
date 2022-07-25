from qtpy.QtWidgets import QFormLayout, QLabel

from ert.gui.ertwidgets import addHelpToWidget
from ert.gui.ertwidgets.caseselector import CaseSelector
from ert.gui.ertwidgets.models.activerealizationsmodel import ActiveRealizationsModel
from ert_shared.models import SingleTestRun
from ert.gui.simulation.simulation_config_panel import SimulationConfigPanel

from ert_shared.libres_facade import LibresFacade

from dataclasses import dataclass


@dataclass
class Arguments:
    mode: str


class SingleTestRunPanel(SimulationConfigPanel):
    def __init__(self, ert, notifier):
        self.ert = ert
        facade = LibresFacade(ert)
        SimulationConfigPanel.__init__(self, SingleTestRun)
        self.setObjectName("Single_test_run_panel")
        layout = QFormLayout()

        case_selector = CaseSelector(facade, notifier)
        layout.addRow("Current case:", case_selector)

        run_path_label = QLabel(
            f"<b>{self.ert.getModelConfig().getRunpathAsString()}</b>"
        )
        addHelpToWidget(run_path_label, "config/simulation/runpath")
        layout.addRow("Runpath:", run_path_label)

        self._active_realizations_model = ActiveRealizationsModel(facade)

        self.setLayout(layout)

    def getSimulationArguments(self):
        return Arguments("test_run")
