from pathlib import Path
from unittest.mock import Mock, patch

import pytest
from PyQt6.QtCore import Qt, QTimer
from PyQt6.QtWidgets import QComboBox, QToolButton, QWidget
from pytestqt.qtbot import QtBot

from ert.config import ErtConfig
from ert.gui.experiments import ExperimentPanel, RunDialog
from ert.gui.experiments.ensemble_experiment_panel import EnsembleExperimentPanel
from ert.gui.main import _setup_main_window
from ert.gui.tools.event_viewer.panel import GUILogHandler
from ert.run_models.ensemble_experiment import EnsembleExperiment
from ert.services import ErtClient, ErtServerController, SharedClient
from tests.ert.handle_runpath_dialog import handle_runpath_dialog


@pytest.fixture
def runpath_case(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    config_path = tmp_path / "config.ert"
    config_path.write_text("NUM_REALIZATIONS 1\nRUNPATH simulations/<IENS>/<ITER>\n")
    config = ErtConfig.from_file(config_path)
    (tmp_path / "simulations" / "0" / "0").mkdir(parents=True)
    SharedClient.close_client()
    with ErtServerController.init_service(project=Path(config.ens_path)):
        try:
            yield config
        finally:
            SharedClient.close_client()


@pytest.mark.slow
def test_that_runpaths_are_preserved_when_deletion_fails(
    runpath_case: ErtConfig, qtbot: QtBot
):
    snake_oil_case = runpath_case
    args_mock = Mock()
    args_mock.config = "snake_oil.ert"

    gui = _setup_main_window(
        snake_oil_case, args_mock, GUILogHandler(), snake_oil_case.ens_path
    )
    experiment_panel = gui.findChild(ExperimentPanel)

    assert isinstance(experiment_panel, ExperimentPanel)
    simulation_mode_combo = experiment_panel.findChild(QComboBox)
    assert isinstance(simulation_mode_combo, QComboBox)
    simulation_mode_combo.setCurrentText(EnsembleExperiment.name())
    simulation_settings = gui.findChild(EnsembleExperimentPanel)
    simulation_settings._experiment_name_field.setText("new_experiment_name")

    # Click start simulation and agree to the message
    run_experiment = experiment_panel.findChild(QWidget, name="run_experiment")
    assert run_experiment
    assert isinstance(run_experiment, QToolButton)

    # Add something to the runpath
    runpath = Path(
        snake_oil_case.runpath_config.runpath_format_string.replace(
            "<IENS>", "0"
        ).replace("<ITER>", "0")
    )
    dummy_file = runpath / "dummy"
    dummy_file.touch()

    QTimer.singleShot(
        1000, lambda: handle_runpath_dialog(gui, qtbot, expect_error=True)
    )
    with patch.object(ErtClient, "runpath_delete", return_value=False) as delete:
        qtbot.mouseClick(run_experiment, Qt.MouseButton.LeftButton)

        qtbot.waitUntil(lambda: gui.findChild(RunDialog) is not None)
        delete.assert_called_once()
    run_dialog = gui.findChild(RunDialog)
    qtbot.waitUntil(lambda: run_dialog.is_experiment_done() is True, timeout=100000)
    qtbot.waitUntil(lambda: run_dialog._tab_widget.currentWidget() is not None)
    assert dummy_file.exists()


@pytest.mark.slow
def test_that_runpaths_are_deleted_when_confirmed(
    runpath_case: ErtConfig, qtbot: QtBot
):
    snake_oil_case = runpath_case
    args_mock = Mock()
    args_mock.config = "snake_oil.ert"

    gui = _setup_main_window(
        snake_oil_case, args_mock, GUILogHandler(), snake_oil_case.ens_path
    )
    experiment_panel = gui.findChild(ExperimentPanel)

    assert isinstance(experiment_panel, ExperimentPanel)
    simulation_mode_combo = experiment_panel.findChild(QComboBox)
    assert isinstance(simulation_mode_combo, QComboBox)
    simulation_mode_combo.setCurrentText(EnsembleExperiment.name())
    simulation_settings = gui.findChild(EnsembleExperimentPanel)
    simulation_settings._experiment_name_field.setText("new_experiment_name")

    # Click start simulation and agree to the message
    run_experiment = experiment_panel.findChild(QWidget, name="run_experiment")
    assert run_experiment
    assert isinstance(run_experiment, QToolButton)

    runpath = Path(
        snake_oil_case.runpath_config.runpath_format_string.replace(
            "<IENS>", "0"
        ).replace("<ITER>", "0")
    )
    dummy_file = runpath / "dummy"
    dummy_file.touch()

    QTimer.singleShot(
        1000, lambda: handle_runpath_dialog(gui, qtbot, delete_runpath=True)
    )
    qtbot.mouseClick(run_experiment, Qt.MouseButton.LeftButton)

    qtbot.waitUntil(lambda: gui.findChild(RunDialog) is not None)
    run_dialog = gui.findChild(RunDialog)
    qtbot.waitUntil(lambda: run_dialog.is_experiment_done() is True, timeout=100000)
    qtbot.waitUntil(lambda: run_dialog._tab_widget.currentWidget() is not None)
    assert not dummy_file.exists()


@pytest.mark.slow
def test_that_runpaths_are_preserved_when_deletion_is_unchecked(
    runpath_case: ErtConfig, qtbot: QtBot
):
    snake_oil_case = runpath_case
    args_mock = Mock()
    args_mock.config = "snake_oil.ert"

    gui = _setup_main_window(
        snake_oil_case, args_mock, GUILogHandler(), snake_oil_case.ens_path
    )
    experiment_panel = gui.findChild(ExperimentPanel)

    assert isinstance(experiment_panel, ExperimentPanel)
    simulation_mode_combo = experiment_panel.findChild(QComboBox)
    assert isinstance(simulation_mode_combo, QComboBox)
    simulation_mode_combo.setCurrentText(EnsembleExperiment.name())
    simulation_settings = gui.findChild(EnsembleExperimentPanel)
    simulation_settings._experiment_name_field.setText("new_experiment_name")

    # Click start simulation and agree to the message
    run_experiment = experiment_panel.findChild(QWidget, name="run_experiment")
    assert run_experiment
    assert isinstance(run_experiment, QToolButton)

    runpath = Path(
        snake_oil_case.runpath_config.runpath_format_string.replace("<IENS>", "0")
    ).parent
    dummy_file = runpath / "dummy"
    dummy_file.touch()

    QTimer.singleShot(
        500, lambda: handle_runpath_dialog(gui, qtbot, delete_runpath=False)
    )
    qtbot.mouseClick(run_experiment, Qt.MouseButton.LeftButton)

    qtbot.waitUntil(lambda: gui.findChild(RunDialog) is not None, timeout=10000)
    run_dialog = gui.findChild(RunDialog)
    qtbot.waitUntil(lambda: run_dialog.is_experiment_done() is True, timeout=100000)
    qtbot.waitUntil(lambda: run_dialog._tab_widget.currentWidget() is not None)
    assert dummy_file.exists()
