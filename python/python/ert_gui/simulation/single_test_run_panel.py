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

    def toggleAdvancedOptions(self, show_advanced):
        pass
