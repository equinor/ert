import os
from pathlib import Path
from unittest.mock import Mock

from pytestqt.qtbot import QtBot
from qtpy.QtCore import Qt, QTimer
from qtpy.QtWidgets import QComboBox, QMessageBox, QToolButton, QWidget

from ert.config import ErtConfig
from ert.enkf_main import EnKFMain
from ert.gui.main import _setup_main_window
from ert.gui.main_window import ErtMainWindow
from ert.gui.simulation.run_dialog import RunDialog
from ert.gui.simulation.simulation_panel import SimulationPanel
from ert.gui.tools.event_viewer.panel import GUILogHandler
from ert.run_models.ensemble_experiment import EnsembleExperiment
from ert.services.storage_service import StorageService
from ert.storage import open_storage


def handle_run_path_dialog(gui: ErtMainWindow, qtbot: QtBot, delete_run_path: bool):
    mb = gui.findChild(QMessageBox, "RUN_PATH_WARNING_BOX")

    if mb is not None:
        assert mb
        assert isinstance(mb, QMessageBox)

        if delete_run_path:
            qtbot.mouseClick(mb.checkBox(), Qt.LeftButton)

        qtbot.mouseClick(mb.buttons()[0], Qt.LeftButton)


def test_run_path_is_deleted(snake_oil_case_storage: ErtConfig, qtbot: QtBot):
    snake_oil_case = snake_oil_case_storage
    args_mock = Mock()
    args_mock.config = "snake_oil.ert"

    with StorageService.init_service(
        ert_config=args_mock.config,
        project=os.path.abspath(snake_oil_case.ens_path),
    ), open_storage(snake_oil_case.ens_path, mode="w") as storage:
        gui = _setup_main_window(EnKFMain(snake_oil_case), args_mock, GUILogHandler())
        gui.notifier.set_storage(storage)
        simulation_panel = gui.findChild(SimulationPanel)

        assert isinstance(simulation_panel, SimulationPanel)
        simulation_mode_combo = simulation_panel.findChild(QComboBox)
        assert isinstance(simulation_mode_combo, QComboBox)
        simulation_mode_combo.setCurrentText(EnsembleExperiment.name())

        # Click start simulation and agree to the message
        start_simulation = simulation_panel.findChild(QWidget, name="start_simulation")
        assert start_simulation
        assert isinstance(start_simulation, QToolButton)

        run_path = Path(
            snake_oil_case.model_config.runpath_format_string.replace("<IENS>", "0")
        ).parent
        with open(run_path / "dummy", "w", encoding="utf-8") as dummy_file:
            dummy_file.close()

        def handle_dialog():
            qtbot.waitUntil(lambda: gui.findChild(QMessageBox) is not None)
            message_box = gui.findChild(QMessageBox)
            assert message_box
            qtbot.mouseClick(message_box.buttons()[0], Qt.LeftButton)

            QTimer.singleShot(
                500, lambda: handle_run_path_dialog(gui, qtbot, delete_run_path=True)
            )

        QTimer.singleShot(500, handle_dialog)
        qtbot.mouseClick(start_simulation, Qt.LeftButton)

        qtbot.waitUntil(lambda: gui.findChild(RunDialog) is not None)
        run_dialog = gui.findChild(RunDialog)
        qtbot.mouseClick(run_dialog.show_details_button, Qt.LeftButton)
        qtbot.waitUntil(run_dialog.done_button.isVisible, timeout=100000)
        qtbot.waitUntil(lambda: run_dialog._tab_widget.currentWidget() is not None)
        qtbot.mouseClick(run_dialog.done_button, Qt.LeftButton)

        assert not os.path.exists(run_path / dummy_file.name)


def test_run_path_is_not_deleted(snake_oil_case_storage: ErtConfig, qtbot: QtBot):
    snake_oil_case = snake_oil_case_storage
    args_mock = Mock()
    args_mock.config = "snake_oil.ert"

    with StorageService.init_service(
        ert_config=args_mock.config,
        project=os.path.abspath(snake_oil_case.ens_path),
    ), open_storage(snake_oil_case.ens_path, mode="w") as storage:
        gui = _setup_main_window(EnKFMain(snake_oil_case), args_mock, GUILogHandler())
        gui.notifier.set_storage(storage)
        simulation_panel = gui.findChild(SimulationPanel)

        assert isinstance(simulation_panel, SimulationPanel)
        simulation_mode_combo = simulation_panel.findChild(QComboBox)
        assert isinstance(simulation_mode_combo, QComboBox)
        simulation_mode_combo.setCurrentText(EnsembleExperiment.name())

        # Click start simulation and agree to the message
        start_simulation = simulation_panel.findChild(QWidget, name="start_simulation")
        assert start_simulation
        assert isinstance(start_simulation, QToolButton)

        run_path = Path(
            snake_oil_case.model_config.runpath_format_string.replace("<IENS>", "0")
        ).parent
        with open(run_path / "dummy", "w", encoding="utf-8") as dummy_file:
            dummy_file.close()

        def handle_dialog():
            qtbot.waitUntil(lambda: gui.findChild(QMessageBox) is not None)
            message_box = gui.findChild(QMessageBox)
            assert message_box
            qtbot.mouseClick(message_box.buttons()[0], Qt.LeftButton)

            QTimer.singleShot(
                500, lambda: handle_run_path_dialog(gui, qtbot, delete_run_path=False)
            )

        QTimer.singleShot(500, handle_dialog)
        qtbot.mouseClick(start_simulation, Qt.LeftButton)

        qtbot.waitUntil(lambda: gui.findChild(RunDialog) is not None, timeout=10000)
        run_dialog = gui.findChild(RunDialog)
        qtbot.mouseClick(run_dialog.show_details_button, Qt.LeftButton)
        qtbot.waitUntil(run_dialog.done_button.isVisible, timeout=100000)
        qtbot.waitUntil(lambda: run_dialog._tab_widget.currentWidget() is not None)
        qtbot.mouseClick(run_dialog.done_button, Qt.LeftButton)

        assert os.path.exists(run_path / dummy_file.name)
