from dataclasses import dataclass
from typing import List

from qtpy.QtWidgets import QCheckBox, QFormLayout, QLabel

from ert.gui.ertnotifier import ErtNotifier
from ert.gui.ertwidgets import ActiveLabel, AnalysisModuleEdit, CaseSelector
from ert.gui.ertwidgets.copyablelabel import CopyableLabel
from ert.gui.ertwidgets.models.activerealizationsmodel import ActiveRealizationsModel
from ert.gui.ertwidgets.models.targetcasemodel import TargetCaseModel
from ert.gui.ertwidgets.models.valuemodel import ValueModel
from ert.gui.ertwidgets.stringbox import StringBox
from ert.libres_facade import LibresFacade
from ert.run_models import MultipleDataAssimilation
from ert.validation import (
    NumberListStringArgument,
    ProperNameFormatArgument,
    RangeStringArgument,
)

from .simulation_config_panel import SimulationConfigPanel


@dataclass
class Arguments:
    mode: str
    target_case: str
    realizations: str
    weights: List[float]
    restart_run: bool
    prior_ensemble: str


class MultipleDataAssimilationPanel(SimulationConfigPanel):
    def __init__(self, facade: LibresFacade, notifier: ErtNotifier, ensemble_size: int):
        SimulationConfigPanel.__init__(self, MultipleDataAssimilation)
        self.notifier = notifier

        layout = QFormLayout()
        self.setObjectName("ES_MDA_panel")

        runpath_label = CopyableLabel(text=facade.run_path_stripped)
        layout.addRow("Runpath:", runpath_label)

        number_of_realizations_label = QLabel(f"<b>{ensemble_size}</b>")
        layout.addRow(QLabel("Number of realizations:"), number_of_realizations_label)

        self._target_case_format_model = TargetCaseModel(
            facade, notifier, format_mode=True
        )
        self._target_case_format_field = StringBox(
            self._target_case_format_model,
            self._target_case_format_model.getDefaultValue(),
            True,
        )
        self._target_case_format_field.setValidator(ProperNameFormatArgument())
        layout.addRow("Target case format:", self._target_case_format_field)

        self.weights = MultipleDataAssimilation.default_weights
        self.weights_valid = True
        self._createInputForWeights(layout)

        self._analysis_module_edit = AnalysisModuleEdit(
            facade.get_analysis_module("STD_ENKF"), ensemble_size
        )
        layout.addRow("Analysis module:", self._analysis_module_edit)

        self._active_realizations_model = ActiveRealizationsModel(ensemble_size)
        self._active_realizations_field = StringBox(
            self._active_realizations_model, "config/simulation/active_realizations"
        )
        self._active_realizations_field.setValidator(RangeStringArgument(ensemble_size))
        layout.addRow("Active realizations:", self._active_realizations_field)

        self._restart_box = QCheckBox("")
        self._restart_box.setObjectName("restart_checkbox_esmda")
        self._restart_box.toggled.connect(self.restart_run_toggled)
        self._restart_box.setEnabled(False)
        layout.addRow("Restart run:", self._restart_box)

        self._case_selector = CaseSelector(notifier)
        self._case_selector.case_populated.connect(self.restart_run_toggled)
        layout.addRow("Restart from:", self._case_selector)

        self._target_case_format_field.getValidationSupport().validationChanged.connect(  # noqa
            self.simulationConfigurationChanged
        )
        self._active_realizations_field.getValidationSupport().validationChanged.connect(  # noqa
            self.simulationConfigurationChanged
        )
        self._relative_iteration_weights_box.getValidationSupport().validationChanged.connect(  # noqa
            self.simulationConfigurationChanged
        )

        self.setLayout(layout)

    def restart_run_toggled(self):
        self._restart_box.setEnabled(bool(self._case_selector._case_list()))
        self._case_selector.setEnabled(self._restart_box.isChecked())

    def _createInputForWeights(self, layout):
        relative_iteration_weights_model = ValueModel(self.weights)
        self._relative_iteration_weights_box = StringBox(
            relative_iteration_weights_model,
            continuous_update=True,
        )
        self._relative_iteration_weights_box.setValidator(NumberListStringArgument())
        layout.addRow("Relative weights:", self._relative_iteration_weights_box)

        relative_iteration_weights_model.valueChanged.connect(self.setWeights)

        normalized_weights_model = ValueModel()
        normalized_weights_widget = ActiveLabel(normalized_weights_model)
        layout.addRow("Normalized weights:", normalized_weights_widget)

        def updateVisualizationOfNormalizedWeights():
            self.weights_valid = False

            if self._relative_iteration_weights_box.isValid():
                weights = MultipleDataAssimilation.parseWeights(
                    relative_iteration_weights_model.getValue()
                )
                normalized_weights = MultipleDataAssimilation.normalizeWeights(weights)
                normalized_weights_model.setValue(
                    ", ".join(f"{x:.2f}" for x in normalized_weights)
                )

                if not weights:
                    normalized_weights_model.setValue("The weights are invalid!")
                else:
                    self.weights_valid = True
            else:
                normalized_weights_model.setValue("The weights are invalid!")

        self._relative_iteration_weights_box.getValidationSupport().validationChanged.connect(  # noqa
            updateVisualizationOfNormalizedWeights
        )

        updateVisualizationOfNormalizedWeights()  # To normalize the default weights

    def isConfigurationValid(self):
        return (
            self._target_case_format_field.isValid()
            and self._active_realizations_field.isValid()
            and self._relative_iteration_weights_box.isValid()
            and self.weights_valid
        )

    def getSimulationArguments(self):
        return Arguments(
            mode="es_mda",
            target_case=self._target_case_format_model.getValue(),
            realizations=self._active_realizations_field.text(),
            weights=self.weights,
            restart_run=self._restart_box.isChecked(),
            prior_ensemble=self._case_selector.currentText(),
        )

    def setWeights(self, weights):
        self.weights = str(weights)
