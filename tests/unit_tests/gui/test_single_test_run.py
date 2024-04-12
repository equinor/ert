import contextlib
import shutil
from datetime import datetime

from qtpy.QtCore import Qt
from qtpy.QtWidgets import (
    QComboBox,
    QWidget,
)

from ert.gui.simulation.run_dialog import RunDialog
from ert.gui.simulation.simulation_panel import SimulationPanel
from ert.run_models import SingleTestRun

from .conftest import get_child, wait_for_child


def test_single_test_run_after_ensemble_experiment(
    ensemble_experiment_has_run_no_failure, qtbot
):
    gui = ensemble_experiment_has_run_no_failure

    # Select correct experiment in the simulation panel
    simulation_panel = get_child(gui, SimulationPanel)
    simulation_mode_combo = get_child(simulation_panel, QComboBox)
    simulation_mode_combo.setCurrentText(SingleTestRun.name())
    # Avoids run path dialog
    with contextlib.suppress(FileNotFoundError):
        shutil.rmtree("poly_out")

    start_simulation = get_child(simulation_panel, QWidget, name="start_simulation")
    qtbot.mouseClick(start_simulation, Qt.LeftButton)
    # The Run dialog opens, click show details and wait until done appears
    # then click it
    run_dialog = wait_for_child(gui, qtbot, RunDialog)
    qtbot.mouseClick(run_dialog.show_details_button, Qt.LeftButton)
    qtbot.waitUntil(run_dialog.done_button.isVisible, timeout=100000)
    qtbot.waitUntil(lambda: run_dialog._tab_widget.currentWidget() is not None)
    qtbot.mouseClick(run_dialog.done_button, Qt.LeftButton)

    storage = gui.notifier.storage
    assert "single_test_run" in [exp.name for exp in storage.experiments]
    assert any(str(datetime.now().year) in ens.name for ens in storage.ensembles)
