import contextlib
import shutil

from PyQt6.QtCore import Qt
from PyQt6.QtWidgets import QComboBox, QWidget

from ert.gui.experiments.experiment_panel import ExperimentPanel
from ert.gui.experiments.run_dialog import RunDialog
from ert.run_models import SingleTestRun

from .conftest import get_child, wait_for_child


def test_single_test_run_after_ensemble_experiment(
    ensemble_experiment_has_run_no_failure, qtbot
):
    gui = ensemble_experiment_has_run_no_failure

    # Select correct experiment in the simulation panel
    experiment_panel = get_child(gui, ExperimentPanel)
    simulation_mode_combo = get_child(experiment_panel, QComboBox)
    simulation_mode_combo.setCurrentText(SingleTestRun.name())
    # Avoids run path dialog
    with contextlib.suppress(FileNotFoundError):
        shutil.rmtree("poly_out")

    simulation_mode_combo = experiment_panel.findChild(QComboBox)
    simulation_mode_combo.setCurrentText("Single realization test-run")

    run_experiment = get_child(experiment_panel, QWidget, name="run_experiment")
    qtbot.mouseClick(run_experiment, Qt.MouseButton.LeftButton)
    run_dialog = wait_for_child(gui, qtbot, RunDialog)
    qtbot.waitUntil(lambda: run_dialog.is_simulation_done() is True, timeout=100000)
    qtbot.waitUntil(lambda: run_dialog._tab_widget.currentWidget() is not None)

    storage = gui.notifier.storage
    assert "single_test_run" in [exp.name for exp in storage.experiments]
    ensemble_names = [ens.name for ens in storage.ensembles]
    assert "iter-0" in ensemble_names
    # Default ensemble name when running single test run
    assert "ensemble" in ensemble_names
