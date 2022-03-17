from qtpy.QtWidgets import QFormLayout, QLabel, QSpinBox

from ert_gui.ertnotifier import ErtNotifier
from ert_gui.ertwidgets import addHelpToWidget, AnalysisModuleSelector, CaseSelector
from ert_gui.ertwidgets.models.activerealizationsmodel import ActiveRealizationsModel
from ert_gui.ertwidgets.models.targetcasemodel import TargetCaseModel
from ert_gui.ertwidgets.stringbox import StringBox
from ert_shared.ide.keywords.definitions import (
    RangeStringArgument,
    ProperNameFormatArgument,
)
from ert_gui.simulation import SimulationConfigPanel
from ert_shared.models import IteratedEnsembleSmoother
from ert_shared.libres_facade import LibresFacade


class IteratedEnsembleSmootherPanel(SimulationConfigPanel):
    def __init__(self, facade: LibresFacade, notifier: ErtNotifier):
        self.facade = facade
        self.notifier = notifier
        SimulationConfigPanel.__init__(self, IteratedEnsembleSmoother)

        layout = QFormLayout()

        case_selector = CaseSelector(self.facade, notifier)
        layout.addRow("Current case:", case_selector)

        run_path_label = QLabel("<b>%s</b>" % self.facade.run_path)
        addHelpToWidget(run_path_label, "config/simulation/runpath")
        layout.addRow("Runpath:", run_path_label)

        number_of_realizations_label = QLabel(
            "<b>%d</b>" % self.facade.get_ensemble_size()
        )
        addHelpToWidget(
            number_of_realizations_label, "config/ensemble/num_realizations"
        )
        layout.addRow(QLabel("Number of realizations:"), number_of_realizations_label)

        # The num_iterations_spinner does not track any external changes (will that ever happen?)
        num_iterations_spinner = QSpinBox()
        num_iterations_spinner.setMinimum(1)
        num_iterations_spinner.setMaximum(100)
        num_iterations_spinner.setValue(self.facade.get_number_of_iterations())
        addHelpToWidget(
            num_iterations_spinner, "config/simulation/number_of_iterations"
        )
        num_iterations_spinner.valueChanged[int].connect(self.setNumberIterations)

        layout.addRow("Number of iterations:", num_iterations_spinner)

        self._iterated_target_case_format_model = TargetCaseModel(
            self.facade, notifier, format_mode=True
        )
        self._iterated_target_case_format_field = StringBox(
            self._iterated_target_case_format_model,
            "config/simulation/iterated_target_case_format",
        )
        self._iterated_target_case_format_field.setValidator(ProperNameFormatArgument())
        layout.addRow("Target case format:", self._iterated_target_case_format_field)

        self._analysis_module_selector = AnalysisModuleSelector(
            self.facade,
            iterable=True,
            help_link="config/analysis/analysis_module",
        )
        layout.addRow("Analysis Module:", self._analysis_module_selector)

        self._active_realizations_model = ActiveRealizationsModel(self.facade)
        self._active_realizations_field = StringBox(
            self._active_realizations_model, "config/simulation/active_realizations"
        )
        self._active_realizations_field.setValidator(
            RangeStringArgument(self.facade.get_ensemble_size())
        )
        layout.addRow("Active realizations", self._active_realizations_field)

        self._iterated_target_case_format_field.getValidationSupport().validationChanged.connect(
            self.simulationConfigurationChanged
        )
        self._active_realizations_field.getValidationSupport().validationChanged.connect(
            self.simulationConfigurationChanged
        )

        self.setLayout(layout)

    def setNumberIterations(self, iteration_count):
        if iteration_count != self.facade.get_number_of_iterations():
            self.facade.get_analysis_config().getAnalysisIterConfig().setNumIterations(
                iteration_count
            )
            self.notifier.emitErtChange()

    def isConfigurationValid(self):
        analysis_module = self._analysis_module_selector.getSelectedAnalysisModuleName()
        return (
            self._iterated_target_case_format_field.isValid()
            and self._active_realizations_field.isValid()
            and analysis_module is not None
        )

    def getSimulationArguments(self):
        arguments = {
            "active_realizations": self._active_realizations_model.getActiveRealizationsMask(),
            "target_case": self._iterated_target_case_format_model.getValue(),
            "analysis_module": self._analysis_module_selector.getSelectedAnalysisModuleName(),
        }
        return arguments
