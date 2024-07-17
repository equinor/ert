import os
import random
import stat
from textwrap import dedent
from typing import Set

import pytest
from qtpy.QtCore import Qt, QTimer
from qtpy.QtWidgets import QCheckBox, QComboBox, QDialogButtonBox, QMessageBox, QWidget

from ert.gui.simulation.experiment_panel import ExperimentPanel
from ert.gui.simulation.run_dialog import RunDialog
from ert.gui.simulation.view import RealizationWidget

from .conftest import get_child, wait_for_child


def test_restart_failed_realizations(opened_main_window_clean, qtbot):
    """This runs an ensemble experiment with some failing realizations, and then
    restarts two times, checking that only the failed realizations are started.
    """
    gui = opened_main_window_clean

    experiment_panel = gui.findChild(ExperimentPanel)

    # Select correct experiment in the simulation panel
    assert isinstance(experiment_panel, ExperimentPanel)
    simulation_mode_combo = experiment_panel.findChild(QComboBox)
    assert isinstance(simulation_mode_combo, QComboBox)
    simulation_mode_combo.setCurrentIndex(1)

    assert simulation_mode_combo.currentText() == "Ensemble experiment"

    # Click start simulation and agree to the message
    run_experiment = experiment_panel.findChild(QWidget, name="run_experiment")
    qtbot.mouseClick(run_experiment, Qt.MouseButton.LeftButton)

    qtbot.waitUntil(lambda: gui.findChild(RunDialog) is not None)
    run_dialog = gui.findChild(RunDialog)

    qtbot.waitUntil(run_dialog.done_button.isVisible, timeout=60000)

    qtbot.mouseClick(run_dialog.done_button, Qt.MouseButton.LeftButton)
    simulation_mode_combo.setCurrentIndex(4)

    assert (
        simulation_mode_combo.currentText()
        == "Multiple Data Assimilation (ES MDA) - Recommended"
    )
    es_mda_panel = gui.findChild(QWidget, name="ES_MDA_panel")
    assert es_mda_panel
    restart_button = es_mda_panel.findChild(QCheckBox, name="restart_checkbox_esmda")
    assert restart_button
    restart_button.click()

    def handle_dialog():
        qtbot.waitUntil(lambda: gui.findChild(QMessageBox) is not None)
        confirm_restart_dialog = gui.findChild(QMessageBox)
        assert isinstance(confirm_restart_dialog, QMessageBox)
        dialog_buttons = confirm_restart_dialog.findChild(QDialogButtonBox).buttons()
        yes_button = [b for b in dialog_buttons if "Yes" in b.text()][0]
        qtbot.mouseClick(yes_button, Qt.LeftButton)

    print(es_mda_panel._ensemble_selector.selected_ensemble.name)
    QTimer.singleShot(500, handle_dialog)
    run_experiment = experiment_panel.findChild(QWidget, name="run_experiment")
    qtbot.mouseClick(run_experiment, Qt.MouseButton.LeftButton)
