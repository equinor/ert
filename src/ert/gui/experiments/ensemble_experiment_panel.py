from dataclasses import dataclass
from typing import override

from PyQt6.QtCore import Qt
from PyQt6.QtCore import pyqtSlot as Slot
from PyQt6.QtWidgets import (
    QFormLayout,
    QLabel,
    QWidget,
)

from ert.config import AnalysisConfig, ParameterConfig
from ert.gui.ertnotifier import ErtNotifier
from ert.gui.ertwidgets import (
    CopyableLabel,
    StringBox,
    TextModel,
)
from ert.mode_definitions import ENSEMBLE_EXPERIMENT_MODE
from ert.run_models import EnsembleExperiment
from ert.validation import ProperNameArgument

from ._panel_utils import (
    add_parameter_configuration_rows,
    create_active_realizations_field,
    create_experiment_name_field,
    create_number_of_realizations_container,
    merge_design_matrix_parameters,
)
from .experiment_config_panel import ExperimentConfigPanel


@dataclass
class Arguments:
    mode: str
    realizations: str
    current_ensemble: str
    experiment_name: str


class EnsembleExperimentPanel(ExperimentConfigPanel):
    def __init__(
        self,
        analysis_config: AnalysisConfig,
        parameter_configuration: list[ParameterConfig],
        active_realizations: list[bool],
        config_num_realization: int,
        runpath: str,
        notifier: ErtNotifier,
    ) -> None:
        super().__init__(EnsembleExperiment)
        self.notifier = notifier
        self.setObjectName("Ensemble_experiment_panel")

        layout = QFormLayout()
        layout.setFormAlignment(Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignTop)

        self._experiment_name_field = create_experiment_name_field(
            self.notifier.storage, ENSEMBLE_EXPERIMENT_MODE
        )
        layout.addRow("Experiment name:", self._experiment_name_field)

        self._ensemble_name_field = StringBox(
            TextModel(""), placeholder_text="ensemble"
        )
        self._ensemble_name_field.setValidator(ProperNameArgument())
        self._ensemble_name_field.setMinimumWidth(250)

        layout.addRow("Ensemble name:", self._ensemble_name_field)

        runpath_label = CopyableLabel(text=runpath)
        layout.addRow("Runpath:", runpath_label)

        number_of_realizations_container, number_of_realizations_label = (
            create_number_of_realizations_container(len(active_realizations))
        )

        layout.addRow(
            QLabel("Number of realizations:"), number_of_realizations_container
        )

        self._active_realizations_field, _ = create_active_realizations_field(
            active_realizations
        )
        layout.addRow("Active realizations", self._active_realizations_field)

        parameter_configuration = merge_design_matrix_parameters(
            analysis_config, parameter_configuration
        )
        add_parameter_configuration_rows(
            self,
            layout,
            analysis_config,
            parameter_configuration,
            number_of_realizations_label=number_of_realizations_label,
            config_num_realization=config_num_realization,
        )
        self.setLayout(layout)
        self._connect_signals()

    def _connect_signals(self) -> None:
        self._active_realizations_field.getValidationSupport().validationChanged.connect(
            self.experiment_configuration_changed
        )
        self._experiment_name_field.getValidationSupport().validationChanged.connect(
            self.experiment_configuration_changed
        )
        self._ensemble_name_field.getValidationSupport().validationChanged.connect(
            self.experiment_configuration_changed
        )
        self.notifier.ertChanged.connect(self._update_experiment_name_placeholder)

    @override
    @Slot(QWidget)
    def experimentTypeChanged(self, w: QWidget) -> None:
        if isinstance(w, EnsembleExperimentPanel):
            self._update_experiment_name_placeholder()

    def _update_experiment_name_placeholder(self) -> None:
        self._experiment_name_field.setPlaceholderText(
            self.notifier.storage.get_unique_experiment_name(ENSEMBLE_EXPERIMENT_MODE)
        )

    @override
    def isConfigurationValid(self) -> bool:
        return (
            self._active_realizations_field.isValid()
            and self._experiment_name_field.isValid()
            and self._ensemble_name_field.isValid()
        )

    @override
    def get_experiment_arguments(self) -> Arguments:
        return Arguments(
            mode=ENSEMBLE_EXPERIMENT_MODE,
            current_ensemble=self._ensemble_name_field.get_text,
            realizations=self._active_realizations_field.text(),
            experiment_name=self._experiment_name_field.get_text,
        )
