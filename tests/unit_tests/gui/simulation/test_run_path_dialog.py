import os
from pathlib import Path
from unittest.mock import Mock, patch

import pytest
from pytestqt.qtbot import QtBot
from qtpy.QtCore import Qt, QTimer
from qtpy.QtWidgets import QComboBox, QMessageBox, QToolButton, QWidget

from ert.config import ErtConfig
from ert.gui.main import _setup_main_window
from ert.gui.main_window import ErtMainWindow
from ert.gui.simulation.ensemble_experiment_panel import EnsembleExperimentPanel
from ert.gui.simulation.experiment_panel import ExperimentPanel
from ert.gui.simulation.run_dialog import RunDialog
from ert.gui.tools.event_viewer.panel import GUILogHandler
from ert.run_models.ensemble_experiment import EnsembleExperiment
from ert.services.storage_service import StorageService
from ert.storage import open_storage


def handle_run_path_dialog(
    gui: ErtMainWindow,
    qtbot: QtBot,
    delete_run_path: bool = True,
    expect_error: bool = False,
):
    mb = gui.findChild(QMessageBox, "RUN_PATH_WARNING_BOX")

    if mb is not None:
        assert mb
        assert isinstance(mb, QMessageBox)

        if delete_run_path:
            qtbot.mouseClick(mb.checkBox(), Qt.LeftButton)

        qtbot.mouseClick(mb.buttons()[0], Qt.LeftButton)
        if expect_error:
            QTimer.singleShot(1000, lambda: handle_run_path_error_dialog(gui, qtbot))


def handle_run_path_error_dialog(gui: ErtMainWindow, qtbot: QtBot):
    mb = gui.findChild(QMessageBox, "RUN_PATH_ERROR_BOX")

    if mb is not None:
        assert mb
        assert isinstance(mb, QMessageBox)
        # Continue without deleting the runpath
        qtbot.mouseClick(mb.buttons()[0], Qt.LeftButton)


@pytest.mark.integration_test
def test_run_path_deleted_error(
    snake_oil_case_storage: ErtConfig, qtbot: QtBot, mocker
):
    snake_oil_case = snake_oil_case_storage
    args_mock = Mock()
    args_mock.config = "snake_oil.ert"

    with StorageService.init_service(
        project=os.path.abspath(snake_oil_case.ens_path),
    ), open_storage(snake_oil_case.ens_path, mode="w") as storage:
        gui = _setup_main_window(snake_oil_case, args_mock, GUILogHandler(), storage)
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
        run_path = Path(
            snake_oil_case.model_config.runpath_format_string.replace(
                "<IENS>", "0"
            ).replace("<ITER>", "0")
        )
        with open(run_path / "dummy", "w", encoding="utf-8") as dummy_file:
            dummy_file.close()

        QTimer.singleShot(
            1000, lambda: handle_run_path_dialog(gui, qtbot, expect_error=True)
        )
        with patch("shutil.rmtree", side_effect=PermissionError("Not allowed!")):
            qtbot.mouseClick(run_experiment, Qt.LeftButton)

            qtbot.waitUntil(lambda: gui.findChild(RunDialog) is not None)
        run_dialog = gui.findChild(RunDialog)
        qtbot.waitUntil(run_dialog.done_button.isVisible, timeout=100000)
        qtbot.waitUntil(lambda: run_dialog._tab_widget.currentWidget() is not None)
        qtbot.mouseClick(run_dialog.done_button, Qt.LeftButton)

        assert os.path.exists(run_path / dummy_file.name)


@pytest.mark.integration_test
def test_run_path_is_deleted(snake_oil_case_storage: ErtConfig, qtbot: QtBot):
    snake_oil_case = snake_oil_case_storage
    args_mock = Mock()
    args_mock.config = "snake_oil.ert"

    with StorageService.init_service(
        project=os.path.abspath(snake_oil_case.ens_path),
    ), open_storage(snake_oil_case.ens_path, mode="w") as storage:
        gui = _setup_main_window(snake_oil_case, args_mock, GUILogHandler(), storage)
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

        run_path = Path(
            snake_oil_case.model_config.runpath_format_string.replace(
                "<IENS>", "0"
            ).replace("<ITER>", "0")
        )
        with open(run_path / "dummy", "w", encoding="utf-8") as dummy_file:
            dummy_file.close()

        QTimer.singleShot(
            1000, lambda: handle_run_path_dialog(gui, qtbot, delete_run_path=True)
        )
        qtbot.mouseClick(run_experiment, Qt.LeftButton)

        qtbot.waitUntil(lambda: gui.findChild(RunDialog) is not None)
        run_dialog = gui.findChild(RunDialog)
        qtbot.waitUntil(run_dialog.done_button.isVisible, timeout=100000)
        qtbot.waitUntil(lambda: run_dialog._tab_widget.currentWidget() is not None)
        qtbot.mouseClick(run_dialog.done_button, Qt.LeftButton)

        assert not os.path.exists(run_path / dummy_file.name)


@pytest.mark.integration_test
def test_run_path_is_not_deleted(snake_oil_case_storage: ErtConfig, qtbot: QtBot):
    snake_oil_case = snake_oil_case_storage
    args_mock = Mock()
    args_mock.config = "snake_oil.ert"

    with StorageService.init_service(
        project=os.path.abspath(snake_oil_case.ens_path),
    ), open_storage(snake_oil_case.ens_path, mode="w") as storage:
        gui = _setup_main_window(snake_oil_case, args_mock, GUILogHandler(), storage)
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

        run_path = Path(
            snake_oil_case.model_config.runpath_format_string.replace("<IENS>", "0")
        ).parent
        with open(run_path / "dummy", "w", encoding="utf-8") as dummy_file:
            dummy_file.close()

        QTimer.singleShot(
            500, lambda: handle_run_path_dialog(gui, qtbot, delete_run_path=False)
        )
        qtbot.mouseClick(run_experiment, Qt.LeftButton)

        qtbot.waitUntil(lambda: gui.findChild(RunDialog) is not None, timeout=10000)
        run_dialog = gui.findChild(RunDialog)
        qtbot.waitUntil(run_dialog.done_button.isVisible, timeout=100000)
        qtbot.waitUntil(lambda: run_dialog._tab_widget.currentWidget() is not None)
        qtbot.mouseClick(run_dialog.done_button, Qt.LeftButton)

        assert os.path.exists(run_path / dummy_file.name)
