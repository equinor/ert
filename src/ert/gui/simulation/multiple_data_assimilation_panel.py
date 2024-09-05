from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, List

from qtpy.QtCore import Slot
from qtpy.QtGui import QFont
from qtpy.QtWidgets import QCheckBox, QFormLayout, QLabel

from ert.gui.ertnotifier import ErtNotifier
from ert.gui.ertwidgets import (
    ActiveRealizationsModel,
    AnalysisModuleEdit,
    CopyableLabel,
    EnsembleSelector,
    StringBox,
    TargetEnsembleModel,
    TextModel,
    ValueModel,
)
from ert.mode_definitions import ES_MDA_MODE
from ert.run_models import MultipleDataAssimilation
from ert.validation import (
    NumberListStringArgument,
    ProperNameFormatArgument,
    RangeStringArgument,
)
from ert.validation.proper_name_argument import ExperimentValidation

from .experiment_config_panel import ExperimentConfigPanel

if TYPE_CHECKING:
    from ert.config import AnalysisConfig
    from ert.gui.ertwidgets import ValueModel


@dataclass
class Arguments:
    mode: str
    target_ensemble: str
    realizations: str
    weights: List[float]
    restart_run: bool
    prior_ensemble_id: str  # UUID not serializable in json
    experiment_name: str


class MultipleDataAssimilationPanel(ExperimentConfigPanel):
    def __init__(
        self,
        analysis_config: AnalysisConfig,
        run_path: str,
        notifier: ErtNotifier,
        ensemble_size: int,
    ) -> None:
        ExperimentConfigPanel.__init__(self, MultipleDataAssimilation)
        self.notifier = notifier

        layout = QFormLayout()
        self.setObjectName("ES_MDA_panel")

        self._experiment_name_field = StringBox(
            TextModel(""),
            placeholder_text=self.notifier.storage.get_unique_experiment_name(
                ES_MDA_MODE
            ),
        )
        self._experiment_name_field.setMinimumWidth(250)
        self._experiment_name_field.setValidator(
            ExperimentValidation(self.notifier.storage)
        )
        self._experiment_name_field.setObjectName("experiment_field")
        layout.addRow("Experiment name:", self._experiment_name_field)

        runpath_label = CopyableLabel(text=run_path)
        layout.addRow("Runpath:", runpath_label)

        number_of_realizations_label = QLabel(f"<b>{ensemble_size}</b>")
        layout.addRow(QLabel("Number of realizations:"), number_of_realizations_label)

        self._target_ensemble_format_model = TargetEnsembleModel(
            analysis_config, notifier
        )
        self._target_ensemble_format_field = StringBox(
            self._target_ensemble_format_model,  # type: ignore
            self._target_ensemble_format_model.getDefaultValue(),  # type: ignore
            continuous_update=True,
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
            self._active_realizations_model,  # type: ignore
            "config/simulation/active_realizations",
        )
        self._active_realizations_field.setValidator(RangeStringArgument(ensemble_size))
        self._ensemble_selector = EnsembleSelector(notifier)
        self._realizations_from_fs()
        layout.addRow("Active realizations:", self._active_realizations_field)

        self._restart_box = QCheckBox("")
        self._restart_box.setObjectName("restart_checkbox_esmda")
        self._restart_box.toggled.connect(self.restart_run_toggled)
        self._restart_box.toggled.connect(self.update_experiment_edit)

        self._restart_box.setEnabled(bool(self._ensemble_selector._ensemble_list()))
        layout.addRow("Restart run:", self._restart_box)

        self._ensemble_selector.ensemble_populated.connect(self.restart_run_toggled)
        self._ensemble_selector.currentIndexChanged.connect(self._realizations_from_fs)
        self._ensemble_selector.currentIndexChanged.connect(self.update_experiment_name)
        layout.addRow("Restart from:", self._ensemble_selector)

        self._experiment_name_field.getValidationSupport().validationChanged.connect(
            self.simulationConfigurationChanged
        )
        self._target_ensemble_format_field.getValidationSupport().validationChanged.connect(
            self.simulationConfigurationChanged
        )
        self._active_realizations_field.getValidationSupport().validationChanged.connect(
            self.simulationConfigurationChanged
        )
        self._relative_iteration_weights_box.getValidationSupport().validationChanged.connect(
            self.simulationConfigurationChanged
        )

        self.setLayout(layout)

        self.notifier.ertChanged.connect(self._update_experiment_name_placeholder)

    @Slot(ExperimentConfigPanel)
    def experimentTypeChanged(self, w: ExperimentConfigPanel) -> None:
        if isinstance(w, MultipleDataAssimilationPanel):
            self._update_experiment_name_placeholder()

    def _update_experiment_name_placeholder(self) -> None:
        self._experiment_name_field.setPlaceholderText(
            self.notifier.storage.get_unique_experiment_name(ES_MDA_MODE)
        )

    @Slot()
    def update_experiment_name(self) -> None:
        if not self._experiment_name_field.isEnabled():
            self._experiment_name_field.setText(
                self._ensemble_selector.selected_ensemble.experiment.name
            )

            self._relative_iteration_weights_box.setText(
                self._ensemble_selector.selected_ensemble.relative_weights
                or MultipleDataAssimilation.default_weights
            )
            self._evaluate_weights_box_enabled()

    @Slot(bool)
    def update_experiment_edit(self, checked: bool) -> None:
        self._experiment_name_field.clear()
        self._experiment_name_field.enable_validation(not checked)
        self._experiment_name_field.setEnabled(not checked)
        if checked:
            self._experiment_name_field.setText(
                self._ensemble_selector.selected_ensemble.experiment.name
            )

        self._evaluate_weights_box_enabled()

    def _evaluate_weights_box_enabled(self) -> None:
        self._relative_iteration_weights_box.setEnabled(
            not self._restart_box.isChecked()
            or not self._ensemble_selector.selected_ensemble.relative_weights
        )

    def restart_run_toggled(self) -> None:
        self._restart_box.setEnabled(bool(self._ensemble_selector._ensemble_list()))
        self._ensemble_selector.setEnabled(self._restart_box.isChecked())

        self._relative_iteration_weights_box.setText(
            self._ensemble_selector.selected_ensemble.relative_weights
            or MultipleDataAssimilation.default_weights
            if self._restart_box.isChecked()
            else MultipleDataAssimilation.default_weights
        )

    def _createInputForWeights(self, layout: QFormLayout) -> None:
        relative_iteration_weights_model = ValueModel(self.weights)
        self._relative_iteration_weights_box = StringBox(
            relative_iteration_weights_model,  # type: ignore
            continuous_update=True,
        )
        self._relative_iteration_weights_box.setObjectName("weights_input_esmda")
        self._relative_iteration_weights_box.setValidator(NumberListStringArgument())
        layout.addRow("Relative weights:", self._relative_iteration_weights_box)

        relative_iteration_weights_model.valueChanged.connect(self.setWeights)

        normalized_weights_model = ValueModel()
        normalized_weights_widget = _ActiveLabel(normalized_weights_model)
        layout.addRow("Normalized weights:", normalized_weights_widget)

        def updateVisualizationOfNormalizedWeights() -> None:
            self.weights_valid = False

            if self._relative_iteration_weights_box.isValid():
                try:
                    normalized_weights = MultipleDataAssimilation.parse_weights(
                        relative_iteration_weights_model.getValue()  # type: ignore
                    )
                    normalized_weights_model.setValue(
                        ", ".join(f"{x:.2f}" for x in normalized_weights)
                    )
                    self.weights_valid = True
                except ValueError:
                    normalized_weights_model.setValue("The weights are invalid!")
            else:
                normalized_weights_model.setValue("The weights are invalid!")

        self._relative_iteration_weights_box.getValidationSupport().validationChanged.connect(
            updateVisualizationOfNormalizedWeights
        )

        updateVisualizationOfNormalizedWeights()  # To normalize the default weights

    def isConfigurationValid(self) -> bool:
        return (
            self._experiment_name_field.isValid()
            and self._target_ensemble_format_field.isValid()
            and self._active_realizations_field.isValid()
            and self._relative_iteration_weights_box.isValid()
            and self.weights_valid
        )

    def get_experiment_arguments(self) -> Arguments:
        return Arguments(
            mode=ES_MDA_MODE,
            target_ensemble=self._target_ensemble_format_model.getValue(),  # type: ignore
            realizations=self._active_realizations_field.text(),
            weights=self.weights,  # type: ignore
            restart_run=self._restart_box.isChecked(),
            prior_ensemble_id=(
                str(self._ensemble_selector.selected_ensemble.id)
                if self._restart_box.isChecked()
                else ""
            ),
            experiment_name=self._experiment_name_field.get_text,
        )

    def setWeights(self, weights: Any) -> None:
        self.weights = str(weights)

    def _realizations_from_fs(self) -> None:
        ensemble = self._ensemble_selector.selected_ensemble
        if ensemble:
            mask = ensemble.get_realization_mask_with_parameters()
            self._active_realizations_field.model.setValueFromMask(mask)  # type: ignore


class _ActiveLabel(QLabel):
    def __init__(self, model: ValueModel) -> None:
        QLabel.__init__(self)

        self._model = model

        font = self.font()
        font.setWeight(QFont.Bold)
        self.setFont(font)

        self._model.valueChanged.connect(self.updateLabel)

        self.updateLabel()

    def updateLabel(self) -> None:
        """Retrieves data from the model and inserts it into the edit line"""
        model_value = self._model.getValue()
        if model_value is None:
            model_value = ""

        self.setText(str(model_value))
