from qtpy.QtWidgets import QFormLayout, QLabel

from ert_gui.ertwidgets import addHelpToWidget
from ert_gui.ertwidgets.caseselector import CaseSelector
from ert_gui.ertwidgets.models.activerealizationsmodel import ActiveRealizationsModel
from ert_gui.ertwidgets.models.ertmodel import getRunPath
from ert_shared.models import SingleTestRun
from ert_gui.simulation.simulation_config_panel import SimulationConfigPanel
from ecl.util.util import BoolVector


class SingleTestRunPanel(SimulationConfigPanel):
    def __init__(self):
        SimulationConfigPanel.__init__(self, SingleTestRun)
        self.setObjectName("Single_test_run_panel")
        layout = QFormLayout()

        case_selector = CaseSelector()
        layout.addRow("Current case:", case_selector)

        run_path_label = QLabel("<b>%s</b>" % getRunPath())
        addHelpToWidget(run_path_label, "config/simulation/runpath")
        layout.addRow("Runpath:", run_path_label)

        self._active_realizations_model = ActiveRealizationsModel()

        self.setLayout(layout)

    def getSimulationArguments(self):

        return {"active_realizations": BoolVector(default_value=True, initial_size=1)}
