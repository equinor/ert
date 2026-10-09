from dataclasses import dataclass
from typing import override

from PyQt6.QtCore import Qt
from PyQt6.QtWidgets import QFormLayout, QLabel

from ert.config import AnalysisConfig, ParameterConfig
from ert.gui.ertnotifier import ErtNotifier
from ert.gui.ertwidgets import CopyableLabel
from ert.mode_definitions import TEST_RUN_MODE
from ert.run_models import SingleTestRun

from ._panel_utils import (
    add_parameter_configuration_rows,
    merge_design_matrix_parameters,
)
from .experiment_config_panel import ExperimentConfigPanel


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

        layout = QFormLayout()
        lab = QLabel(" ".join(SingleTestRun.__doc__.split()))  # type: ignore
        lab.setWordWrap(True)
        lab.setAlignment(Qt.AlignmentFlag.AlignLeft)
        layout.addRow(lab)

        runpath_label = CopyableLabel(text=runpath)
        layout.addRow("Runpath:", runpath_label)

        parameter_configuration = merge_design_matrix_parameters(
            analysis_config.design_matrix, parameter_configuration
        )
        add_parameter_configuration_rows(
            self,
            layout,
            analysis_config.design_matrix,
            lambda: parameter_configuration,
        )
        self.setLayout(layout)

    @override
    def get_experiment_arguments(self) -> Arguments:
        return Arguments(TEST_RUN_MODE, "ensemble", "single_test_run")
