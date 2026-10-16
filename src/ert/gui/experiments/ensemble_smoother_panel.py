from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, override

from PyQt6.QtCore import Qt
from PyQt6.QtCore import pyqtSlot as Slot
from PyQt6.QtWidgets import (
    QApplication,
    QFormLayout,
    QHBoxLayout,
    QLabel,
    QStyle,
    QVBoxLayout,
    QWidget,
)

from ert.config import ParameterConfig
from ert.config.parameter_config import has_updatable_parameters
from ert.gui.ertnotifier import ErtNotifier
from ert.gui.ertwidgets import (
    AnalysisModuleEdit,
    CopyableLabel,
)
from ert.gui.ertwidgets.models.parameter_configuration_state_model import (
    ParameterConfigurationStateModel,
)
from ert.mode_definitions import ENSEMBLE_SMOOTHER_MODE
from ert.run_models import EnsembleSmoother
from ert.run_models.ensemble_smoother import DEPRECATION_MESSAGE

from ._panel_utils import (
    add_parameter_configuration_rows,
    create_active_realizations_field,
    create_experiment_name_field,
    create_number_of_realizations_container,
    create_target_ensemble_format_field,
    merge_design_matrix_parameters,
)
from ._update_strategy_summary_widget import UpdateStrategySummaryWidget
from .experiment_config_panel import ExperimentConfigPanel

if TYPE_CHECKING:
    from ert.config import AnalysisConfig


@dataclass
class Arguments:
    mode: str
    target_ensemble: str
    realizations: str
    experiment_name: str
    parameter_configuration: list[ParameterConfig]


def _create_deprecation_banner() -> QWidget:
    banner = QWidget()
    layout = QHBoxLayout(banner)
    layout.setContentsMargins(0, 0, 0, 0)

    icon = QLabel()
    style = QApplication.style()
    if style is not None:
        icon.setPixmap(
            style.standardIcon(QStyle.StandardPixmap.SP_MessageBoxWarning).pixmap(
                16, 16
            )
        )

    warning = QLabel(
        f"{DEPRECATION_MESSAGE} Select 'Multiple data assimilation' "
        "and check 'Single update'."
    )
    warning.setWordWrap(True)
    warning.setAlignment(Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignTop)
    warning.setObjectName("ensemble_smoother_deprecation_warning")
    layout.addWidget(icon, alignment=Qt.AlignmentFlag.AlignTop)
    layout.addWidget(warning, stretch=1)
    return banner


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

        self._param_state = ParameterConfigurationStateModel(
            merge_design_matrix_parameters(
                analysis_config.design_matrix, parameter_configuration
            ),
            self,
        )
        self._analysis_module_edit = AnalysisModuleEdit(
            es_settings=analysis_config.es_settings,
            get_update_strategies=lambda: self._param_state.update_strategies,
            ensemble_size=sum(
                active_realizations
            ),  # only use active realizations for setting threshold
        )
        self._analysis_module_edit.setObjectName("ensemble_smoother_edit")
        self._analysis_module_edit.update_strategies_changed.connect(
            self._param_state.apply_update_strategies
        )

        layout.addRow("Update settings:", self._analysis_module_edit)
        self._update_strategy_label = QLabel("Parameter Localizations")
        self._update_strategy_label.setObjectName("update_strategy_label")
        self._update_strategy_summary_widget = UpdateStrategySummaryWidget(
            self._param_state, self
        )
        self._update_strategy_label.setToolTip(
            self._update_strategy_summary_widget.toolTip()
        )
        layout.addRow(self._update_strategy_label, self._update_strategy_summary_widget)

        self._active_realizations_field, _ = create_active_realizations_field(
            active_realizations
        )
        layout.addRow("Active realizations", self._active_realizations_field)

        add_parameter_configuration_rows(
            self,
            layout,
            analysis_config.design_matrix,
            lambda: self._param_state.parameters,
            number_of_realizations_label=number_of_realizations_label,
            config_num_realization=config_num_realization,
        )
        panel_layout = QVBoxLayout()
        panel_layout.addWidget(_create_deprecation_banner())
        panel_layout.addLayout(layout)
        panel_layout.addStretch()
        self.setLayout(panel_layout)
        self._connect_signals()

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
            and has_updatable_parameters(self._param_state.parameters)
        )

    @override
    def get_experiment_arguments(self) -> Arguments:
        return Arguments(
            mode=ENSEMBLE_SMOOTHER_MODE,
            target_ensemble=self._ensemble_format_model.getValue(),  # type: ignore
            realizations=self._active_realizations_field.text(),
            experiment_name=self._experiment_name_field.get_text,
            parameter_configuration=self._param_state.parameters,
        )
