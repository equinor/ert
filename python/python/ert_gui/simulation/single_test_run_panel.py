from PyQt4.QtGui import QFormLayout, QLabel

from ert_gui.ertwidgets import addHelpToWidget
from ert_gui.ertwidgets.caseselector import CaseSelector
from ert_gui.ertwidgets.models.activerealizationsmodel import ActiveRealizationsModel
from ert_gui.ertwidgets.models.ertmodel import getRealizationCount, getRunPath
from ert_gui.ertwidgets.stringbox import StringBox
from ert_gui.ide.keywords.definitions import RangeStringArgument
from ert_gui.simulation.models import EnsembleExperiment
from ert_gui.simulation.models import SingleTestRun
from ert_gui.simulation.simulation_config_panel import SimulationConfigPanel

class SingleTestRunPanel(SimulationConfigPanel):

    def __init__(self):
        SimulationConfigPanel.__init__(self, SingleTestRun())
        
        layout = QFormLayout()

        case_selector = CaseSelector()
        layout.addRow("Current case:", case_selector)

        run_path_label = QLabel("<b>%s</b>" % getRunPath())
        addHelpToWidget(run_path_label, "config/simulation/runpath")
        layout.addRow("Runpath:", run_path_label)

        self._active_realizations_model = ActiveRealizationsModel()

        self.setLayout(layout)

    def toggleAdvancedOptions(self, show_advanced):
        pass

    def getSimulationArguments(self):
        active_realizations_mask = self._active_realizations_model.getActiveRealizationsMask()
        return {"active_realizations": active_realizations_mask}


