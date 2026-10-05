from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, override

from PyQt6.QtCore import Qt
from PyQt6.QtCore import pyqtSlot as Slot
from PyQt6.QtWidgets import QFormLayout, QLabel, QWidget

from ert.config.parameter_config import has_updatable_parameters
from ert.gui.ertnotifier import ErtNotifier
from ert.gui.ertwidgets import (
    AnalysisModuleEdit,
    CopyableLabel,
    get_parameters_button,
)
from ert.mode_definitions import ENSEMBLE_SMOOTHER_MODE
from ert.run_models import EnsembleSmoother

from ._design_matrix_panel import DesignMatrixPanel
from ._panel_helpers import (
    create_active_realizations_field,
    create_experiment_name_field,
    create_number_of_realizations_container,
    create_target_ensemble_format_field,
)
from .experiment_config_panel import ExperimentConfigPanel

if TYPE_CHECKING:
    from ert.config import AnalysisConfig, ParameterConfig


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
        parameter_configuration: list[ParameterConfig],
        runpath: str,
        notifier: ErtNotifier,
        active_realizations: list[bool],
        config_num_realization: int,
    ) -> None:
        super().__init__(EnsembleSmoother)
        self.notifier = notifier
        self.setObjectName("ensemble_smoother_panel")

        layout = QFormLayout()
        layout.setFormAlignment(Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignTop)

        self._experiment_name_field = create_experiment_name_field(
            self.notifier.storage, ENSEMBLE_SMOOTHER_MODE
        )
        layout.addRow("Experiment name:", self._experiment_name_field)

        runpath_label = CopyableLabel(text=runpath)
        layout.addRow("Runpath:", runpath_label)

        number_of_realizations_container, number_of_realizations_label = (
            create_number_of_realizations_container(len(active_realizations))
        )

        layout.addRow(
            QLabel("Number of realizations:"), number_of_realizations_container
        )

        (
            self._ensemble_format_model,
            self._ensemble_format_field,
        ) = create_target_ensemble_format_field(analysis_config, notifier)
        layout.addRow("Ensemble format:", self._ensemble_format_field)

        self._analysis_module_edit = AnalysisModuleEdit(
            es_settings=analysis_config.es_settings,
            parameter_config=parameter_configuration,
            ensemble_size=sum(
                active_realizations
            ),  # only use active realizations for setting threshold
        )
        self._analysis_module_edit.setObjectName("ensemble_smoother_edit")

        layout.addRow("Update settings:", self._analysis_module_edit)
        self._active_realizations_field, _ = create_active_realizations_field(
            active_realizations
        )
        layout.addRow("Active realizations", self._active_realizations_field)

        self._add_parameter_configuration_rows(
            layout,
            analysis_config,
            number_of_realizations_label,
            config_num_realization,
        )
        self.setLayout(layout)
        self._connect_signals()

    def _add_parameter_configuration_rows(
        self,
        layout: QFormLayout,
        analysis_config: AnalysisConfig,
        number_of_realizations_label: QLabel,
        config_num_realization: int,
    ) -> None:
        design_matrix = analysis_config.design_matrix
        if design_matrix is not None:
            layout.addRow(
                "Design matrix",
                DesignMatrixPanel.get_design_matrix_button(
                    design_matrix,
                    number_of_realizations_label,
                    config_num_realization,
                ),
            )
            self._analysis_module_edit.parameter_config = (
                design_matrix.merge_with_existing_parameters(
                    self._analysis_module_edit.parameter_config
                )
            )

        if self._analysis_module_edit.parameter_config:
            layout.addRow(
                "Parameters",
                get_parameters_button(
                    self._analysis_module_edit.parameter_config, self
                ),
            )

    def _connect_signals(self) -> None:
        self._experiment_name_field.getValidationSupport().validationChanged.connect(
            self.experiment_configuration_changed
        )
        self._ensemble_format_field.getValidationSupport().validationChanged.connect(
            self.experiment_configuration_changed
        )
        self._active_realizations_field.getValidationSupport().validationChanged.connect(
            self.experiment_configuration_changed
        )
        self.notifier.ertChanged.connect(self._update_experiment_name_placeholder)

    @override
    @Slot(QWidget)
    def experimentTypeChanged(self, w: QWidget) -> None:
        if isinstance(w, EnsembleSmootherPanel):
            self._update_experiment_name_placeholder()

    def _update_experiment_name_placeholder(self) -> None:
        self._experiment_name_field.setPlaceholderText(
            self.notifier.storage.get_unique_experiment_name(ENSEMBLE_SMOOTHER_MODE)
        )

    @override
    def isConfigurationValid(self) -> bool:
        return (
            self._experiment_name_field.isValid()
            and self._ensemble_format_field.isValid()
            and self._active_realizations_field.isValid()
            and has_updatable_parameters(self._analysis_module_edit.parameter_config)
        )

    @override
    def get_experiment_arguments(self) -> Arguments:
        return Arguments(
            mode=ENSEMBLE_SMOOTHER_MODE,
            target_ensemble=self._ensemble_format_model.getValue(),  # type: ignore
            realizations=self._active_realizations_field.text(),
            experiment_name=self._experiment_name_field.get_text,
        )
