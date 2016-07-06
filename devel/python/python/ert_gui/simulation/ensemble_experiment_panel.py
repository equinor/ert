from PyQt4.QtGui import QFormLayout, QLabel

from ert_gui.ertwidgets import addHelpToWidget
from ert_gui.ertwidgets.caseselector import CaseSelector
from ert_gui.ertwidgets.models.ertmodel import getRealizationCount, getRunPath
from ert_gui.ide.keywords.definitions import RangeStringArgument
from ert_gui.models.connectors.run import EnsembleExperiment, ActiveRealizationsModel
from ert_gui.simulation.simulation_config_panel import SimulationConfigPanel
from ert_gui.widgets.string_box import StringBox


class EnsembleExperimentPanel(SimulationConfigPanel):

    def __init__(self):
        SimulationConfigPanel.__init__(self, EnsembleExperiment())

        layout = QFormLayout()

        case_selector = CaseSelector()
        layout.addRow("Current case:", case_selector)

        run_path_label = QLabel("<b>%s</b>" % getRunPath())
        addHelpToWidget(run_path_label, "config/simulation/runpath")
        layout.addRow("Runpath:", run_path_label)

        number_of_realizations_label = QLabel("<b>%d</b>" % getRealizationCount())
        addHelpToWidget(number_of_realizations_label, "config/ensemble/num_realizations")
        layout.addRow(QLabel("Number of realizations:"), number_of_realizations_label)

        active_realizations_model = ActiveRealizationsModel()
        self.active_realizations_field = StringBox(active_realizations_model, "Active realizations", "config/simulation/active_realizations")
        self.active_realizations_field.setValidator(RangeStringArgument(getRealizationCount()))
        layout.addRow(self.active_realizations_field.getLabel(), self.active_realizations_field)

        self.active_realizations_field.validationChanged.connect(self.simulationConfigurationChanged)

        self.setLayout(layout)


    def isConfigurationValid(self):
        return self.active_realizations_field.isValid()

    def toggleAdvancedOptions(self, show_advanced):
        self.active_realizations_field.setVisible(show_advanced)
        self.layout().labelForField(self.active_realizations_field).setVisible(show_advanced)
