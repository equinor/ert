import logging
import os
import random
import stat
from pathlib import Path
from textwrap import dedent

from PyQt6.QtCore import Qt, QTimer
from PyQt6.QtWidgets import QComboBox, QMessageBox, QWidget

from ert.gui.ertwidgets import StringBox
from ert.gui.simulation.experiment_panel import ExperimentPanel
from ert.gui.simulation.run_dialog import RunDialog
from ert.gui.simulation.view import RealizationWidget

from .conftest import wait_for_child


def test_rerun_failed_all_realizations(opened_main_window_poly, qtbot):
    """This runs an ensemble experiment where all realizations fails, and then
    restarts, checking that all realizations are started.
    """
    gui = opened_main_window_poly

    def write_poly_eval(failing_reals: bool):
        Path("poly_eval.py").write_text(
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
                        if {failing_reals}:
                            sys.exit(1)
                        coeffs = _load_coeffs("parameters.json")
                        output = [_evaluate(coeffs, x) for x in range(10)]
                        with open("poly.out", "w", encoding="utf-8") as f:
                            f.write("\\n".join(map(str, output)))
                    """
            ),
            encoding="utf-8",
        )

        os.chmod(
            "poly_eval.py",
            os.stat("poly_eval.py").st_mode
            | stat.S_IXUSR
            | stat.S_IXGRP
            | stat.S_IXOTH,
        )

    write_poly_eval(failing_reals=True)

    experiment_panel = gui.findChild(ExperimentPanel)

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
    qtbot.waitUntil(lambda: run_dialog.is_simulation_done() is True, timeout=60000)
    qtbot.waitUntil(lambda: run_dialog._tab_widget.currentWidget() is not None)

    run_model = opened_main_window_poly._experiment_panel._model
    # Check that all realizations failed
    assert all(run_model._create_mask_from_failed_realizations())

    def handle_dialog():
        message_box = gui.findChildren(QMessageBox, name="restart_prompt")[-1]
        qtbot.mouseClick(message_box.buttons()[0], Qt.MouseButton.LeftButton)

    write_poly_eval(failing_reals=False)
    QTimer.singleShot(500, handle_dialog)
    qtbot.mouseClick(run_dialog.rerun_button, Qt.MouseButton.LeftButton)

    qtbot.waitUntil(lambda: run_dialog.is_simulation_done() is True, timeout=60000)
    qtbot.waitUntil(lambda: run_dialog._tab_widget.currentWidget() is not None)

    assert not any(run_model._create_mask_from_failed_realizations()), (
        "Not all realizations were successfull"
    )


def test_rerun_failed_realizations(opened_main_window_poly, qtbot, caplog):
    """This runs an ensemble experiment with some failing realizations, and then
    restarts two times, checking that only the failed realizations are started.
    Verifies that the number of successful and failed realizations is logged correctly
    """
    gui = opened_main_window_poly
    caplog.set_level(logging.INFO)

    def write_poly_eval(failing_reals: set[int]):
        Path("poly_eval.py").write_text(
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
                    """  # noqa: E501
            ),
            encoding="utf-8",
        )

        os.chmod(
            "poly_eval.py",
            os.stat("poly_eval.py").st_mode
            | stat.S_IXUSR
            | stat.S_IXGRP
            | stat.S_IXOTH,
        )

    experiment_panel = gui.findChild(ExperimentPanel)
    num_reals = experiment_panel.config.runpath_config.num_realizations

    failing_reals_first_try = {*random.sample(range(num_reals), 5)}
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
    qtbot.waitUntil(lambda: run_dialog.is_simulation_done() is True, timeout=60000)
    qtbot.waitUntil(lambda: run_dialog._tab_widget.currentWidget() is not None)

    def verify_logged_realization_status(realization_count: int, failed_count: int):
        expected_success = realization_count - failed_count
        assert f"number of realizations succeeding: {expected_success}" in caplog.text
        assert f"number of realizations failing: {failed_count}" in caplog.text

    verify_logged_realization_status(num_reals, len(failing_reals_first_try))

    # Assert that the number of boxes in the detailed view is
    # equal to the number of realizations
    realization_widget = run_dialog._tab_widget.currentWidget()
    assert isinstance(realization_widget, RealizationWidget)
    list_model = realization_widget._real_view.model()
    assert list_model
    assert (
        list_model.rowCount() == experiment_panel.config.runpath_config.num_realizations
    )

    run_model = opened_main_window_poly._experiment_panel._model
    # Check we have failed realizations
    assert any(run_model._create_mask_from_failed_realizations())
    failed_realizations = [
        i
        for i, mask in enumerate(run_model._create_mask_from_failed_realizations())
        if mask
    ]

    assert set(failed_realizations) == failing_reals_first_try

    def handle_dialog():
        message_box = gui.findChildren(QMessageBox, name="restart_prompt")[-1]
        qtbot.mouseClick(message_box.buttons()[0], Qt.MouseButton.LeftButton)

    failing_reals_second_try = {*random.sample(list(failing_reals_first_try), 3)}
    write_poly_eval(failing_reals=failing_reals_second_try)
    QTimer.singleShot(500, handle_dialog)
    qtbot.mouseClick(run_dialog.rerun_button, Qt.MouseButton.LeftButton)

    qtbot.waitUntil(lambda: run_dialog.is_simulation_done() is True, timeout=60000)
    qtbot.waitUntil(lambda: run_dialog._tab_widget.currentWidget() is not None)

    verify_logged_realization_status(
        len(failing_reals_first_try), len(failing_reals_second_try)
    )

    # We expect to have the same amount of realizations in list_model
    # since we reuse the snapshot_model
    realization_widget = run_dialog._tab_widget.currentWidget()
    assert isinstance(realization_widget, RealizationWidget)
    list_model = realization_widget._real_view.model()
    assert list_model
    assert (
        list_model.rowCount() == experiment_panel.config.runpath_config.num_realizations
    )

    # Second restart
    assert any(run_model._create_mask_from_failed_realizations())
    failed_realizations = [
        i
        for i, mask in enumerate(run_model._create_mask_from_failed_realizations())
        if mask
    ]
    assert set(failed_realizations) == (
        failing_reals_second_try.union(failing_reals_second_try)
    )

    QTimer.singleShot(500, handle_dialog)
    failing_reals_third_try = {*random.sample(list(failing_reals_second_try), 2)}
    write_poly_eval(failing_reals=failing_reals_third_try)
    qtbot.mouseClick(run_dialog.rerun_button, Qt.MouseButton.LeftButton)

    qtbot.waitUntil(lambda: run_dialog.is_simulation_done() is True, timeout=60000)
    qtbot.waitUntil(lambda: run_dialog._tab_widget.currentWidget() is not None)

    verify_logged_realization_status(
        len(failing_reals_second_try), len(failing_reals_third_try)
    )

    # We expect to have the same amount of realizations in list_model
    # since we reuse the snapshot_model
    realization_widget = run_dialog._tab_widget.currentWidget()
    assert isinstance(realization_widget, RealizationWidget)
    list_model = realization_widget._real_view.model()
    assert list_model
    assert (
        list_model.rowCount() == experiment_panel.config.runpath_config.num_realizations
    )


def handle_run_path_dialog(
    gui,
    qtbot,
):
    mb = gui.findChildren(QMessageBox, "RUN_PATH_WARNING_BOX")
    mb = mb[-1] if mb else None

    if mb is not None:
        assert mb
        assert isinstance(mb, QMessageBox)

        qtbot.mouseClick(mb.buttons()[0], Qt.MouseButton.LeftButton)


def test_rerun_failed_realizations_evaluate_ensemble(
    ensemble_experiment_has_run_no_failure, qtbot
):
    """This runs an evaluate ensemble with some failing realizations, and then
    restarts and checks that only the failed realizations are running.
    """
    gui = ensemble_experiment_has_run_no_failure

    def write_poly_eval(failing_reals: set[int]):
        Path("poly_eval.py").write_text(
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
                    """  # noqa: E501
            ),
            encoding="utf-8",
        )

        os.chmod(
            "poly_eval.py",
            os.stat("poly_eval.py").st_mode
            | stat.S_IXUSR
            | stat.S_IXGRP
            | stat.S_IXOTH,
        )

    experiment_panel = gui.findChild(ExperimentPanel)
    num_reals = experiment_panel.config.runpath_config.num_realizations

    failing_reals_first_try = {*random.sample(range(num_reals), 10)}
    write_poly_eval(failing_reals=failing_reals_first_try)

    # Select correct experiment in the simulation panel
    assert isinstance(experiment_panel, ExperimentPanel)
    simulation_mode_combo = experiment_panel.findChild(QComboBox)
    assert isinstance(simulation_mode_combo, QComboBox)
    simulation_mode_combo.setCurrentText("Evaluate ensemble")

    evaluate_ensemble_panel = gui.findChild(QWidget, name="Evaluate_parameters_panel")
    active_real_field = evaluate_ensemble_panel.findChild(StringBox)
    active_real_field.setText(f"0-{num_reals - 1}")

    # Click start simulation and agree to the message
    run_experiment = experiment_panel.findChild(QWidget, name="run_experiment")

    QTimer.singleShot(1000, lambda: handle_run_path_dialog(gui, qtbot))
    qtbot.mouseClick(run_experiment, Qt.MouseButton.LeftButton)
    # The Run dialog opens, wait until restart appears and the tab is ready
    run_dialog = wait_for_child(gui, qtbot, RunDialog)
    qtbot.waitUntil(lambda: run_dialog.is_simulation_done() is True, timeout=60000)
    qtbot.waitUntil(lambda: run_dialog._tab_widget.currentWidget() is not None)

    # Assert that the number of boxes in the detailed view is
    # equal to the number of realizations
    realization_widget = run_dialog._tab_widget.currentWidget()
    assert isinstance(realization_widget, RealizationWidget)
    list_model = realization_widget._real_view.model()
    assert list_model
    assert (
        list_model.rowCount() == experiment_panel.config.runpath_config.num_realizations
    )

    run_model = gui._experiment_panel._model
    # Check we have failed realizations
    assert any(run_model._create_mask_from_failed_realizations())
    failed_realizations = [
        i
        for i, mask in enumerate(run_model._create_mask_from_failed_realizations())
        if mask
    ]

    assert set(failed_realizations) == failing_reals_first_try

    def handle_dialog():
        message_box = gui.findChildren(QMessageBox, name="restart_prompt")[-1]
        qtbot.mouseClick(message_box.buttons()[0], Qt.MouseButton.LeftButton)

    failing_reals_second_try = {*random.sample(list(failing_reals_first_try), 5)}
    write_poly_eval(failing_reals=failing_reals_second_try)
    QTimer.singleShot(500, handle_dialog)
    qtbot.mouseClick(run_dialog.rerun_button, Qt.MouseButton.LeftButton)

    qtbot.waitUntil(lambda: run_dialog.is_simulation_done() is True, timeout=60000)
    qtbot.waitUntil(lambda: run_dialog._tab_widget.currentWidget() is not None)

    # We expect to have the same amount of realizations in list_model
    # since we reuse the snapshot_model
    realization_widget = run_dialog._tab_widget.currentWidget()
    assert isinstance(realization_widget, RealizationWidget)
    list_model = realization_widget._real_view.model()
    assert list_model
    assert (
        list_model.rowCount() == experiment_panel.config.runpath_config.num_realizations
    )

    # Second restart
    assert any(run_model._create_mask_from_failed_realizations())
    failed_realizations = [
        i
        for i, mask in enumerate(run_model._create_mask_from_failed_realizations())
        if mask
    ]
    assert set(failed_realizations) == failing_reals_second_try
