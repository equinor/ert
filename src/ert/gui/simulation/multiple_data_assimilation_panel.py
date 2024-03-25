from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, List

from qtpy.QtWidgets import QCheckBox, QFormLayout, QLabel, QLineEdit

from ert.gui.ertnotifier import ErtNotifier
from ert.gui.ertwidgets import ActiveLabel, AnalysisModuleEdit, EnsembleSelector
from ert.gui.ertwidgets.copyablelabel import CopyableLabel
from ert.gui.ertwidgets.models.activerealizationsmodel import ActiveRealizationsModel
from ert.gui.ertwidgets.models.targetensemblemodel import TargetEnsembleModel
from ert.gui.ertwidgets.models.valuemodel import ValueModel
from ert.gui.ertwidgets.stringbox import StringBox
from ert.run_models import MultipleDataAssimilation
from ert.validation import (
    NumberListStringArgument,
    ProperNameFormatArgument,
    RangeStringArgument,
)

from .simulation_config_panel import SimulationConfigPanel

if TYPE_CHECKING:
    from ert.config import AnalysisConfig


@dataclass
class Arguments:
    mode: str
    target_ensemble: str
    realizations: str
    weights: List[float]
    restart_run: bool
    prior_ensemble: str
    experiment_name: str


class MultipleDataAssimilationPanel(SimulationConfigPanel):
    def __init__(
        self,
        analysis_config: AnalysisConfig,
        run_path,
        notifier: ErtNotifier,
        ensemble_size: int,
    ):
        SimulationConfigPanel.__init__(self, MultipleDataAssimilation)
        self.notifier = notifier

        layout = QFormLayout()
        self.setObjectName("ES_MDA_panel")

        self._name_field = QLineEdit()
        self._name_field.setPlaceholderText("es_mda")
        self._name_field.setMinimumWidth(250)
        layout.addRow("Experiment name:", self._name_field)

        runpath_label = CopyableLabel(text=run_path)
        layout.addRow("Runpath:", runpath_label)

        number_of_realizations_label = QLabel(f"<b>{ensemble_size}</b>")
        layout.addRow(QLabel("Number of realizations:"), number_of_realizations_label)

        self._target_ensemble_format_model = TargetEnsembleModel(
            analysis_config, notifier
        )
        self._target_ensemble_format_field = StringBox(
            self._target_ensemble_format_model,
            self._target_ensemble_format_model.getDefaultValue(),
            True,
        )
        self._target_ensemble_format_field.setValidator(ProperNameFormatArgument())
        layout.addRow("Target ensemble format:", self._target_ensemble_format_field)

        self.weights = MultipleDataAssimilation.default_weights
        self.weights_valid = True
        self._createInputForWeights(layout)

        self._analysis_module_edit = AnalysisModuleEdit(
            analysis_config.es_module, ensemble_size
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

        self._ensemble_selector = EnsembleSelector(notifier)
        self._ensemble_selector.ensemble_populated.connect(self.restart_run_toggled)
        layout.addRow("Restart from:", self._ensemble_selector)

        self._target_ensemble_format_field.getValidationSupport().validationChanged.connect(  # noqa
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
        self._restart_box.setEnabled(bool(self._ensemble_selector._ensemble_list()))
        self._ensemble_selector.setEnabled(self._restart_box.isChecked())

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
            self._target_ensemble_format_field.isValid()
            and self._active_realizations_field.isValid()
            and self._relative_iteration_weights_box.isValid()
            and self.weights_valid
        )

    def getSimulationArguments(self):
        return Arguments(
            mode="es_mda",
            target_ensemble=self._target_ensemble_format_model.getValue(),
            realizations=self._active_realizations_field.text(),
            weights=self.weights,
            restart_run=self._restart_box.isChecked(),
            prior_ensemble=self._ensemble_selector.currentText(),
            experiment_name=(
                self._name_field.text()
                if self._name_field.text()
                else self._name_field.placeholderText()
            ),
        )

    def setWeights(self, weights):
        self.weights = str(weights)
