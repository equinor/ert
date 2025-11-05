from dataclasses import dataclass

from PyQt6.QtCore import Qt
from PyQt6.QtWidgets import QFormLayout, QLabel
from typing_extensions import override

from ert.gui.ertnotifier import ErtNotifier
from ert.gui.ertwidgets import CopyableLabel
from ert.mode_definitions import TEST_RUN_MODE
from ert.run_models import SingleTestRun

from ...config import AnalysisConfig, ParameterConfig
from ..ertwidgets.parameterviewer import get_parameters_button
from ._design_matrix_panel import DesignMatrixPanel
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
        run_path: str,
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

        runpath_label = CopyableLabel(text=run_path)
        layout.addRow("Runpath:", runpath_label)
        design_matrix = analysis_config.design_matrix
        merged_parameters = parameter_configuration
        if design_matrix is not None:
            layout.addRow(
                "Design Matrix",
                DesignMatrixPanel.get_design_matrix_button(
                    design_matrix,
                ),
            )
            merged_parameters = design_matrix.merge_with_existing_parameters(
                merged_parameters
            )

        if merged_parameters:
            layout.addRow("Parameters", get_parameters_button(merged_parameters, self))

        self.setLayout(layout)

    @override
    def get_experiment_arguments(self) -> Arguments:
        return Arguments(TEST_RUN_MODE, "ensemble", "single_test_run")
