from pytestqt.qtbot import QtBot

from ert.config.analysis_config import AnalysisConfig
from ert.gui.ertnotifier import ErtNotifier
from ert.gui.experiments.single_test_run_panel import SingleTestRunPanel
from ert.mode_definitions import TEST_RUN_MODE


def test_that_experiment_arguments_use_the_test_run_mode_and_fixed_names(
    qtbot: QtBot,
):
    panel = SingleTestRunPanel(AnalysisConfig(), [], "runpath", ErtNotifier())
    qtbot.addWidget(panel)

    arguments = panel.get_experiment_arguments()

    assert arguments.mode == TEST_RUN_MODE
    assert arguments.current_ensemble == "ensemble"
    assert arguments.experiment_name == "single_test_run"
