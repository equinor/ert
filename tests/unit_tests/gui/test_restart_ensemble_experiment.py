import os
import stat
from textwrap import dedent

import pytest
from qtpy.QtCore import Qt, QTimer
from qtpy.QtWidgets import QComboBox, QMessageBox, QWidget

from ert.gui.simulation.run_dialog import RunDialog
from ert.gui.simulation.simulation_panel import SimulationPanel
from ert.gui.simulation.view import RealizationWidget
from tests.unit_tests.gui.simulation.test_run_path_dialog import handle_run_path_dialog

from .conftest import wait_for_child


@pytest.mark.scheduler
def test_restart_failed_realizations(
    opened_main_window_clean, qtbot, try_queue_and_scheduler
):
    """This runs an ensemble experiment with some failing realizations, and then
    does a restart, checking that only the failed realizations are started.
    """
    gui = opened_main_window_clean

    with open("poly_eval.py", "w", encoding="utf-8") as f:
        f.write(
            dedent(
                """\
                #!/usr/bin/env python
                import numpy as np
                import sys
                import json

                def _load_coeffs(filename):
                    with open(filename, encoding="utf-8") as f:
                        return json.load(f)["COEFFS"]

                def _evaluate(coeffs, x):
                    return coeffs["a"] * x**2 + coeffs["b"] * x + coeffs["c"]

                if __name__ == "__main__":
                    if np.random.random(1) > 0.5:
                        sys.exit(1)
                    coeffs = _load_coeffs("parameters.json")
                    output = [_evaluate(coeffs, x) for x in range(10)]
                    with open("poly.out", "w", encoding="utf-8") as f:
                        f.write("\\n".join(map(str, output)))
                """
            )
        )
    os.chmod(
        "poly_eval.py",
        os.stat("poly_eval.py").st_mode | stat.S_IXUSR | stat.S_IXGRP | stat.S_IXOTH,
    )
    # Select correct experiment in the simulation panel
    simulation_panel = gui.findChild(SimulationPanel)
    assert isinstance(simulation_panel, SimulationPanel)
    simulation_mode_combo = simulation_panel.findChild(QComboBox)
    assert isinstance(simulation_mode_combo, QComboBox)
    simulation_mode_combo.setCurrentText("Ensemble experiment")

    # Click start simulation and agree to the message
    start_simulation = simulation_panel.findChild(QWidget, name="start_simulation")

    def handle_dialog():
        qtbot.mouseClick(
            wait_for_child(gui, qtbot, QMessageBox).buttons()[0], Qt.LeftButton
        )

        QTimer.singleShot(
            500, lambda: handle_run_path_dialog(gui, qtbot, delete_run_path=True)
        )

    QTimer.singleShot(500, handle_dialog)
    qtbot.mouseClick(start_simulation, Qt.LeftButton)

    # The Run dialog opens, click show details and wait until done appears
    # then click it
    qtbot.waitUntil(lambda: gui.findChild(RunDialog) is not None)
    run_dialog = gui.findChild(RunDialog)

    qtbot.mouseClick(run_dialog.show_details_button, Qt.LeftButton)

    qtbot.waitUntil(run_dialog.restart_button.isVisible, timeout=200000)
    qtbot.waitUntil(lambda: run_dialog._tab_widget.currentWidget() is not None)

    # Assert that the number of boxes in the detailed view is
    # equal to the number of realizations
    realization_widget = run_dialog._tab_widget.currentWidget()
    assert isinstance(realization_widget, RealizationWidget)
    list_model = realization_widget._real_view.model()
    assert (
        list_model.rowCount()
        == simulation_panel.ert.ert_config.model_config.num_realizations
    )

    # Check we have failed realizations
    assert any(run_dialog._run_model._create_mask_from_failed_realizations())
    failed_realizations = [
        i
        for i, mask in enumerate(
            run_dialog._run_model._create_mask_from_failed_realizations()
        )
        if mask
    ]

    def handle_dialog():
        message_box = wait_for_child(gui, qtbot, QMessageBox, name="restart_prompt")
        qtbot.mouseClick(message_box.buttons()[0], Qt.LeftButton)

    QTimer.singleShot(500, handle_dialog)
    qtbot.mouseClick(run_dialog.restart_button, Qt.LeftButton)

    qtbot.waitUntil(run_dialog.done_button.isVisible, timeout=200000)
    qtbot.waitUntil(lambda: run_dialog._tab_widget.currentWidget() is not None)

    # Assert that the number of boxes in the detailed view is
    # equal to the number of previously failed realizations
    realization_widget = run_dialog._tab_widget.currentWidget()
    assert isinstance(realization_widget, RealizationWidget)
    list_model = realization_widget._real_view.model()
    assert list_model.rowCount() == len(failed_realizations)

    qtbot.mouseClick(run_dialog.done_button, Qt.LeftButton)
    qtbot.mouseClick(run_dialog.done_button, Qt.LeftButton)
