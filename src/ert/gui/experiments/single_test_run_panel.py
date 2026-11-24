from dataclasses import dataclass
from typing import override

from PyQt6.QtCore import Qt
from PyQt6.QtWidgets import QLabel

from ert.config import AnalysisConfig, ParameterConfig
from ert.gui.ertnotifier import ErtNotifier
from ert.mode_definitions import TEST_RUN_MODE
from ert.run_models import SingleTestRun

from ._panel_utils import (
    add_parameter_configuration_rows,
    merge_design_matrix_parameters,
)
from .experiment_config_panel import ExperimentConfigPanel

_EXPERIMENT_NAME = "single_test_run"


@dataclass
class Arguments:
    mode: str
    current_ensemble: str
    experiment_name: str


class SingleTestRunPanel(ExperimentConfigPanel):
    def __init__(
        self,
        analysis_config: AnalysisConfig,
        parameter_configuration: list[ParameterConfig],
        runpath: str,
        notifier: ErtNotifier,
    ) -> None:
        ExperimentConfigPanel.__init__(self, SingleTestRun)
        self.notifier = notifier
        self.setObjectName("Single_test_run_panel")

        layout = self.options_layout
        lab = QLabel(" ".join(SingleTestRun.__doc__.split()))  # type: ignore
        lab.setWordWrap(True)
        lab.setAlignment(Qt.AlignmentFlag.AlignLeft)
        layout.addRow(lab)

        # A test run has no settings to change, so show them read only.
        self.add_common_settings(
            runpath,
            QLabel("<b>1</b>"),
            QLabel("<b>0</b>"),
            experiment_name=QLabel(f"<b>{_EXPERIMENT_NAME}</b>"),
        )

        parameter_configuration = merge_design_matrix_parameters(
            analysis_config.design_matrix, parameter_configuration
        )
        add_parameter_configuration_rows(
            self,
            layout,
            design_matrix=analysis_config.design_matrix,
            get_parameter_configuration=lambda: parameter_configuration,
        )
        self.build_layout(accordions=False)

    @override
    def get_experiment_arguments(self) -> Arguments:
        return Arguments(TEST_RUN_MODE, "ensemble", _EXPERIMENT_NAME)
