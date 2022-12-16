from dataclasses import dataclass

from qtpy.QtWidgets import QFormLayout, QLabel

from ert.gui.ertwidgets import addHelpToWidget
from ert.gui.ertwidgets.caseselector import CaseSelector
from ert.gui.ertwidgets.models.activerealizationsmodel import ActiveRealizationsModel
from ert.libres_facade import LibresFacade
from ert.shared.models import SingleTestRun

from .simulation_config_panel import SimulationConfigPanel


@dataclass
class Arguments:
    mode: str


def escape_string(string):
    return string.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")


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
            f"<b>{escape_string(self.ert.getModelConfig().runpath_format_string)}</b>"
        )
        addHelpToWidget(run_path_label, "config/simulation/runpath")
        layout.addRow("Runpath:", run_path_label)

        self._active_realizations_model = ActiveRealizationsModel(facade)

        self.setLayout(layout)

    def getSimulationArguments(self):
        return Arguments("test_run")
