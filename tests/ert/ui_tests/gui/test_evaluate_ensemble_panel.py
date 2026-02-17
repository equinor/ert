from PyQt6.QtCore import Qt, QTimer
from PyQt6.QtWidgets import QComboBox, QWidget

from ert.gui.experiments import ExperimentPanel, RunDialog
from ert.gui.experiments.evaluate_ensemble_panel import EvaluateEnsemblePanel
from tests.ert.ui_tests.gui.conftest import (
    DEFAULT_NUM_REALIZATIONS,
    get_child,
    wait_for_child,
)
from tests.ert.ui_tests.gui.test_restart_ensemble_experiment import (
    handle_run_path_dialog,
)


def test_evaluate_ensemble_active_realizations_resets_to_all_realizations_with_parameters_when_all_realizations_are_successful(  # noqa
    qtbot, opened_main_window_poly
):
    gui = opened_main_window_poly

    num_reals = DEFAULT_NUM_REALIZATIONS
    expected_active_reals = f"0-{num_reals - 1}" if num_reals > 1 else "0"

    experiment_panel = gui.findChild(ExperimentPanel)
    simulation_mode_combo = experiment_panel.findChild(QComboBox)
    simulation_mode_combo.setCurrentText("Ensemble experiment")

    run_experiment = experiment_panel.findChild(QWidget, name="run_experiment")
    QTimer.singleShot(1000, lambda: handle_run_path_dialog(gui, qtbot))
    qtbot.mouseClick(run_experiment, Qt.MouseButton.LeftButton)

    run_dialog = wait_for_child(gui, qtbot, RunDialog)
    qtbot.waitUntil(lambda: run_dialog.is_simulation_done() is True, timeout=60000)
    qtbot.waitUntil(lambda: run_dialog._tab_widget.currentWidget() is not None)

    evaluate_ensemble_active_realizations = get_child(
        experiment_panel, EvaluateEnsemblePanel
    )._active_realizations_field.model.getValue()

    assert evaluate_ensemble_active_realizations == expected_active_reals
