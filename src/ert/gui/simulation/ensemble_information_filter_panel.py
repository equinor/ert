from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

from PyQt6.QtCore import pyqtSlot as Slot
from PyQt6.QtWidgets import QFormLayout, QLabel, QWidget

from ert.gui.ertnotifier import ErtNotifier
from ert.gui.ertwidgets import (
    ActiveRealizationsModel,
    CopyableLabel,
    StringBox,
    TargetEnsembleModel,
    TextModel,
)
from ert.mode_definitions import ENIF_MODE
from ert.run_models import EnsembleInformationFilter
from ert.validation import (
    ExperimentValidation,
    ProperNameFormatArgument,
)
from ert.validation.active_range import ActiveRange
from ert.validation.range_string_argument import RangeSubsetStringArgument

from ._design_matrix_panel import DesignMatrixPanel
from .experiment_config_panel import ExperimentConfigPanel

if TYPE_CHECKING:
    from ert.config import AnalysisConfig


@dataclass
class Arguments:
    mode: str
    target_ensemble: str
    realizations: str
    experiment_name: str


class EnsembleInformationFilterPanel(ExperimentConfigPanel):
    def __init__(
        self,
        analysis_config: AnalysisConfig,
        run_path: str,
        notifier: ErtNotifier,
        active_realizations: list[bool],
        config_num_realization: int,
    ) -> None:
        super().__init__(EnsembleInformationFilter)
        self.notifier = notifier

        layout = QFormLayout()
        self.setObjectName("enif_panel")

        self._experiment_name_field = StringBox(
            TextModel(""),
            placeholder_text=self.notifier.storage.get_unique_experiment_name(
                ENIF_MODE
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

        number_of_realizations_label = QLabel(f"<b>{len(active_realizations)}</b>")
        layout.addRow(QLabel("Number of realizations:"), number_of_realizations_label)

        self._ensemble_format_model = TargetEnsembleModel(analysis_config, notifier)
        self._ensemble_format_field = StringBox(
            self._ensemble_format_model,  # type: ignore
            self._ensemble_format_model.getDefaultValue(),  # type: ignore
            continuous_update=True,
        )
        self._ensemble_format_field.setValidator(ProperNameFormatArgument())
        layout.addRow("Ensemble format:", self._ensemble_format_field)

        self._active_realizations_field = StringBox(
            ActiveRealizationsModel(len(active_realizations)),  # type: ignore
            "config/simulation/active_realizations",
        )
        self._active_realizations_field.setValidator(
            RangeSubsetStringArgument(ActiveRange(active_realizations)),
        )
        self._active_realizations_field.model.setValueFromMask(  # type: ignore
            active_realizations
        )
        layout.addRow("Active realizations", self._active_realizations_field)

        design_matrix = analysis_config.design_matrix
        if design_matrix is not None:
            layout.addRow(
                "Design Matrix",
                DesignMatrixPanel.get_design_matrix_button(
                    design_matrix,
                    number_of_realizations_label,
                    config_num_realization,
                ),
            )

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

    @Slot(QWidget)
    def experimentTypeChanged(self, w: QWidget) -> None:
        if isinstance(w, EnsembleInformationFilterPanel):
            self._update_experiment_name_placeholder()

    def _update_experiment_name_placeholder(self) -> None:
        self._experiment_name_field.setPlaceholderText(
            self.notifier.storage.get_unique_experiment_name(ENIF_MODE)
        )

    def isConfigurationValid(self) -> bool:
        return (
            self._experiment_name_field.isValid()
            and self._ensemble_format_field.isValid()
            and self._active_realizations_field.isValid()
        )

    def get_experiment_arguments(self) -> Arguments:
        arguments = Arguments(
            mode=ENIF_MODE,
            target_ensemble=self._ensemble_format_model.getValue(),  # type: ignore
            realizations=self._active_realizations_field.text(),
            experiment_name=self._experiment_name_field.get_text,
        )
        return arguments
