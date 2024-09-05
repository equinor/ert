from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

from qtpy.QtCore import Slot
from qtpy.QtWidgets import QFormLayout, QLabel

from ert.gui.ertnotifier import ErtNotifier
from ert.gui.ertwidgets import (
    ActiveRealizationsModel,
    AnalysisModuleEdit,
    CopyableLabel,
    StringBox,
    TargetEnsembleModel,
    TextModel,
)
from ert.mode_definitions import ENSEMBLE_SMOOTHER_MODE
from ert.run_models import EnsembleSmoother
from ert.validation import ProperNameFormatArgument, RangeStringArgument
from ert.validation.proper_name_argument import ExperimentValidation

from .experiment_config_panel import ExperimentConfigPanel

if TYPE_CHECKING:
    from ert.config import AnalysisConfig


@dataclass
class Arguments:
    mode: str
    target_ensemble: str
    realizations: str
    experiment_name: str


class EnsembleSmootherPanel(ExperimentConfigPanel):
    def __init__(
        self,
        analysis_config: AnalysisConfig,
        run_path: str,
        notifier: ErtNotifier,
        ensemble_size: int,
    ) -> None:
        super().__init__(EnsembleSmoother)
        self.notifier = notifier
        layout = QFormLayout()

        self.setObjectName("ensemble_smoother_panel")

        self._experiment_name_field = StringBox(
            TextModel(""),
            placeholder_text=self.notifier.storage.get_unique_experiment_name(
                ENSEMBLE_SMOOTHER_MODE
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

        self._ensemble_format_model = TargetEnsembleModel(analysis_config, notifier)
        self._ensemble_format_field = StringBox(
            self._ensemble_format_model,  # type: ignore
            self._ensemble_format_model.getDefaultValue(),  # type: ignore
            continuous_update=True,
        )
        self._ensemble_format_field.setValidator(ProperNameFormatArgument())
        layout.addRow("Ensemble format:", self._ensemble_format_field)

        self._analysis_module_edit = AnalysisModuleEdit(
            analysis_config.es_module, ensemble_size
        )
        self._analysis_module_edit.setObjectName("ensemble_smoother_edit")
        layout.addRow("Analysis module:", self._analysis_module_edit)

        active_realizations_model = ActiveRealizationsModel(ensemble_size)
        self._active_realizations_field = StringBox(
            active_realizations_model,  # type: ignore
            "config/simulation/active_realizations",
        )
        self._active_realizations_field.setValidator(RangeStringArgument(ensemble_size))
        layout.addRow("Active realizations", self._active_realizations_field)

        self.setLayout(layout)

        self._experiment_name_field.getValidationSupport().validationChanged.connect(
            self.simulationConfigurationChanged
        )
        self._ensemble_format_field.getValidationSupport().validationChanged.connect(
            self.simulationConfigurationChanged
        )
        self._active_realizations_field.getValidationSupport().validationChanged.connect(
            self.simulationConfigurationChanged
        )

        self.notifier.ertChanged.connect(self._update_experiment_name_placeholder)

    @Slot(ExperimentConfigPanel)
    def experimentTypeChanged(self, w: ExperimentConfigPanel) -> None:
        if isinstance(w, EnsembleSmootherPanel):
            self._update_experiment_name_placeholder()

    def _update_experiment_name_placeholder(self) -> None:
        self._experiment_name_field.setPlaceholderText(
            self.notifier.storage.get_unique_experiment_name(ENSEMBLE_SMOOTHER_MODE)
        )

    def isConfigurationValid(self) -> bool:
        return (
            self._experiment_name_field.isValid()
            and self._ensemble_format_field.isValid()
            and self._active_realizations_field.isValid()
        )

    def get_experiment_arguments(self) -> Arguments:
        arguments = Arguments(
            mode=ENSEMBLE_SMOOTHER_MODE,
            target_ensemble=self._ensemble_format_model.getValue(),  # type: ignore
            realizations=self._active_realizations_field.text(),
            experiment_name=self._experiment_name_field.get_text,
        )
        return arguments
