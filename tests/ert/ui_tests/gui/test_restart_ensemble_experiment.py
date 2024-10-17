import os
import random
import stat
from textwrap import dedent
from typing import Set

from qtpy.QtCore import Qt, QTimer
from qtpy.QtWidgets import QComboBox, QMessageBox, QWidget

from ert.gui.simulation.experiment_panel import ExperimentPanel
from ert.gui.simulation.run_dialog import RunDialog
from ert.gui.simulation.view import RealizationWidget

from .conftest import wait_for_child


def test_restart_failed_realizations(opened_main_window_poly, qtbot):
    """This runs an ensemble experiment with some failing realizations, and then
    restarts two times, checking that only the failed realizations are started.
    """
    gui = opened_main_window_poly

    def write_poly_eval(failing_reals: Set[int]):
        with open("poly_eval.py", "w", encoding="utf-8") as f:
            f.write(
                dedent(
                    f"""\
                    #!/usr/bin/env python
                    import numpy as np
                    import sys
                    import json
                    import os

                    def _load_coeffs(filename):
                        with open(filename, encoding="utf-8") as f:
                            return json.load(f)["COEFFS"]

                    def _evaluate(coeffs, x):
                        return coeffs["a"] * x**2 + coeffs["b"] * x + coeffs["c"]

                    if __name__ == "__main__":
                        if int(os.getenv("_ERT_REALIZATION_NUMBER")) in {failing_reals!s}:
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
            os.stat("poly_eval.py").st_mode
            | stat.S_IXUSR
            | stat.S_IXGRP
            | stat.S_IXOTH,
        )

    experiment_panel = gui.findChild(ExperimentPanel)
    num_reals = experiment_panel.config.model_config.num_realizations

    failing_reals_first_try = {*random.sample(range(num_reals), 10)}
    write_poly_eval(failing_reals=failing_reals_first_try)

    # Select correct experiment in the simulation panel
    assert isinstance(experiment_panel, ExperimentPanel)
    simulation_mode_combo = experiment_panel.findChild(QComboBox)
    assert isinstance(simulation_mode_combo, QComboBox)
    simulation_mode_combo.setCurrentText("Ensemble experiment")

    # Click start simulation and agree to the message
    run_experiment = experiment_panel.findChild(QWidget, name="run_experiment")
    qtbot.mouseClick(run_experiment, Qt.MouseButton.LeftButton)

    # The Run dialog opens, wait until restart appears and the tab is ready
    run_dialog = wait_for_child(gui, qtbot, RunDialog)
    qtbot.waitUntil(lambda: not run_dialog.restart_button.isHidden(), timeout=60000)
    qtbot.waitUntil(lambda: run_dialog._tab_widget.currentWidget() is not None)

    # Assert that the number of boxes in the detailed view is
    # equal to the number of realizations
    realization_widget = run_dialog._tab_widget.currentWidget()
    assert isinstance(realization_widget, RealizationWidget)
    list_model = realization_widget._real_view.model()
    assert list_model
    assert (
        list_model.rowCount() == experiment_panel.config.model_config.num_realizations
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

    assert set(failed_realizations) == failing_reals_first_try

    def handle_dialog():
        message_box = wait_for_child(gui, qtbot, QMessageBox, name="restart_prompt")
        qtbot.mouseClick(message_box.buttons()[0], Qt.MouseButton.LeftButton)

    QTimer.singleShot(500, handle_dialog)
    failing_reals_second_try = {*random.sample(list(failing_reals_first_try), 5)}
    write_poly_eval(failing_reals=failing_reals_second_try)
    qtbot.mouseClick(run_dialog.restart_button, Qt.MouseButton.LeftButton)

    qtbot.waitUntil(lambda: not run_dialog.restart_button.isHidden(), timeout=60000)
    qtbot.waitUntil(lambda: run_dialog._tab_widget.currentWidget() is not None)

    # We expect to have the same amount of realizations in list_model
    # since we reuse the snapshot_model
    realization_widget = run_dialog._tab_widget.currentWidget()
    assert isinstance(realization_widget, RealizationWidget)
    list_model = realization_widget._real_view.model()
    assert list_model
    assert (
        list_model.rowCount() == experiment_panel.config.model_config.num_realizations
    )

    # Second restart
    assert any(run_dialog._run_model._create_mask_from_failed_realizations())
    failed_realizations = [
        i
        for i, mask in enumerate(
            run_dialog._run_model._create_mask_from_failed_realizations()
        )
        if mask
    ]
    assert set(failed_realizations) == (
        failing_reals_second_try.union(failing_reals_second_try)
    )

    QTimer.singleShot(500, handle_dialog)
    failing_reals_third_try = {*random.sample(list(failing_reals_second_try), 2)}
    write_poly_eval(failing_reals=failing_reals_third_try)
    qtbot.mouseClick(run_dialog.restart_button, Qt.MouseButton.LeftButton)

    qtbot.waitUntil(lambda: not run_dialog.done_button.isHidden(), timeout=60000)
    qtbot.waitUntil(lambda: run_dialog._tab_widget.currentWidget() is not None)

    # We expect to have the same amount of realizations in list_model
    # since we reuse the snapshot_model
    realization_widget = run_dialog._tab_widget.currentWidget()
    assert isinstance(realization_widget, RealizationWidget)
    list_model = realization_widget._real_view.model()
    assert list_model
    assert (
        list_model.rowCount() == experiment_panel.config.model_config.num_realizations
    )

    qtbot.mouseClick(run_dialog.done_button, Qt.MouseButton.LeftButton)
