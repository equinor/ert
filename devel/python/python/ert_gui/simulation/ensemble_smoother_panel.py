from PyQt4.QtGui import QFormLayout, QLabel

from ert_gui.ertwidgets.caseselector import CaseSelector
from ert_gui.ertwidgets.models.ertmodel import getRealizationCount, getRunPath
from ert_gui.ide.keywords.definitions import RangeStringArgument, ProperNameArgument
from ert_gui.models.connectors.run import ActiveRealizationsModel, TargetCaseModel
from ert_gui.simulation import SimulationConfigPanel
from ert_gui.ertwidgets import addHelpToWidget, AnalysisModuleSelector
from ert_gui.simulation.models import EnsembleSmoother
from ert_gui.widgets.string_box import StringBox


class EnsembleSmootherPanel(SimulationConfigPanel):
    def __init__(self):
        SimulationConfigPanel.__init__(self, EnsembleSmoother())

        layout = QFormLayout()

        case_selector = CaseSelector()
        layout.addRow("Current case:", case_selector)

        run_path_label = QLabel("<b>%s</b>" % getRunPath())
        addHelpToWidget(run_path_label, "config/simulation/runpath")
        layout.addRow("Runpath:", run_path_label)

        number_of_realizations_label = QLabel("<b>%d</b>" % getRealizationCount())
        addHelpToWidget(number_of_realizations_label, "config/ensemble/num_realizations")
        layout.addRow(QLabel("Number of realizations:"), number_of_realizations_label)

        target_case_model = TargetCaseModel()
        self.target_case_field = StringBox(target_case_model, "Target case", "config/simulation/target_case")
        self.target_case_field.setValidator(ProperNameArgument())
        layout.addRow(self.target_case_field.getLabel(), self.target_case_field)


        self._analysis_module_selector = AnalysisModuleSelector(iterable=False, help_link="config/analysis/analysis_module")

        layout.addRow("Analysis Module:", self._analysis_module_selector)

        active_realizations_model = ActiveRealizationsModel()
        self.active_realizations_field = StringBox(active_realizations_model, "Active realizations", "config/simulation/active_realizations")
        self.active_realizations_field.setValidator(RangeStringArgument())
        layout.addRow(self.active_realizations_field.getLabel(), self.active_realizations_field)

        self.target_case_field.validationChanged.connect(self.simulationConfigurationChanged)
        self.active_realizations_field.validationChanged.connect(self.simulationConfigurationChanged)

        self.setLayout(layout)

    def isConfigurationValid(self):
        return self.target_case_field.isValid() and self.active_realizations_field.isValid()

    def toggleAdvancedOptions(self, show_advanced):
        self.active_realizations_field.setVisible(show_advanced)
        self.layout().labelForField(self.active_realizations_field).setVisible(show_advanced)

        self._analysis_module_selector.setVisible(show_advanced)
        self.layout().labelForField(self._analysis_module_selector).setVisible(show_advanced)

    def getSimulationArguments(self):
        return {"analysis_module": self._analysis_module_selector.getSelectedAnalysisModuleName()}
