from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, override

from PyQt6.QtCore import Qt
from PyQt6.QtCore import pyqtSlot as Slot
from PyQt6.QtWidgets import QFormLayout, QLabel, QWidget

from ert.config.parameter_config import has_updatable_parameters
from ert.gui.ertnotifier import ErtNotifier
from ert.gui.ertwidgets import (
    CopyableLabel,
    get_parameters_button,
)
from ert.mode_definitions import ENIF_MODE
from ert.run_models import EnsembleInformationFilter

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


class EnsembleInformationFilterPanel(ExperimentConfigPanel):
    def __init__(
        self,
        analysis_config: AnalysisConfig,
        parameter_configuration: list[ParameterConfig],
        runpath: str,
        notifier: ErtNotifier,
        active_realizations: list[bool],
        config_num_realization: int,
    ) -> None:
        super().__init__(EnsembleInformationFilter)
        self.notifier = notifier

        layout = QFormLayout()
        layout.setFormAlignment(Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignTop)
        self.setObjectName("enif_panel")

        if analysis_config.parameter_type_update_strategies:
            strategies_html = "".join(
                f"<li><b>{param_type}</b>: {strategy.name}</li>"
                for param_type, strategy in (
                    analysis_config.parameter_type_update_strategies.items()
                )
            )
            warning_label = QLabel(
                "<b>Warning:</b> Any given update strategy will be ignored "
                "when running Ensemble Information Filter."
                "<p>The following parameter update strategies were found "
                "in the configuration:"
                f"<ul>{strategies_html}</ul>"
            )
            warning_label.setObjectName("enif_warning_panel")
            warning_label.setWordWrap(True)
            warning_label.setTextFormat(Qt.TextFormat.RichText)
            warning_label.setStyleSheet(
                "QLabel#enif_warning_panel {"
                " background-color: #fff3cd;"
                " color: #664d03;"
                " border: 1px solid #ffe69c;"
                " border-radius: 4px;"
                " padding: 8px;"
                "}"
            )
            layout.addRow(warning_label)

        self._experiment_name_field = create_experiment_name_field(
            self.notifier.storage, ENIF_MODE
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

        self._active_realizations_field, _ = create_active_realizations_field(
            active_realizations
        )
        layout.addRow("Active realizations", self._active_realizations_field)

        self._parameter_configuration = parameter_configuration
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
            self._parameter_configuration = (
                design_matrix.merge_with_existing_parameters(
                    self._parameter_configuration
                )
            )

        if self._parameter_configuration:
            layout.addRow(
                "Parameters", get_parameters_button(self._parameter_configuration, self)
            )

        self.setLayout(layout)

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
        if isinstance(w, EnsembleInformationFilterPanel):
            self._update_experiment_name_placeholder()

    def _update_experiment_name_placeholder(self) -> None:
        self._experiment_name_field.setPlaceholderText(
            self.notifier.storage.get_unique_experiment_name(ENIF_MODE)
        )

    @override
    def isConfigurationValid(self) -> bool:
        return (
            self._experiment_name_field.isValid()
            and self._ensemble_format_field.isValid()
            and self._active_realizations_field.isValid()
            and has_updatable_parameters(self._parameter_configuration)
        )

    @override
    def get_experiment_arguments(self) -> Arguments:
        return Arguments(
            mode=ENIF_MODE,
            target_ensemble=self._ensemble_format_model.getValue(),  # type: ignore
            realizations=self._active_realizations_field.text(),
            experiment_name=self._experiment_name_field.get_text,
        )
