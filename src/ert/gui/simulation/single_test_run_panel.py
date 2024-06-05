from dataclasses import dataclass
from datetime import datetime

from qtpy.QtWidgets import QFormLayout

from ert.gui.ertnotifier import ErtNotifier
from ert.gui.ertwidgets.copyablelabel import CopyableLabel
from ert.mode_definitions import TEST_RUN_MODE
from ert.run_models import SingleTestRun

from .experiment_config_panel import ExperimentConfigPanel


@dataclass
class Arguments:
    mode: str
    current_ensemble: str
    experiment_name: str


class SingleTestRunPanel(ExperimentConfigPanel):
    def __init__(self, run_path: str, notifier: ErtNotifier):
        ExperimentConfigPanel.__init__(self, SingleTestRun)
        self.notifier = notifier
        self.setObjectName("Single_test_run_panel")

        layout = QFormLayout()

        runpath_label = CopyableLabel(text=run_path)
        layout.addRow("Runpath:", runpath_label)

        self.setLayout(layout)

    def get_experiment_arguments(self) -> Arguments:
        ensemble_name = f"{datetime.now().strftime('%Y-%m-%dT%H%M')}"
        return Arguments(TEST_RUN_MODE, ensemble_name, "single_test_run")
