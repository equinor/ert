import argparse
import os
import shutil
from pathlib import Path
from unittest.mock import MagicMock, Mock, PropertyMock, patch

import pytest
from qtpy.QtCore import Qt, QTimer
from qtpy.QtWidgets import QComboBox, QMessageBox, QToolButton, QWidget

import ert.gui
from ert._c_wrappers.enkf import EnKFMain, ErtConfig
from ert.gui.about_dialog import AboutDialog
from ert.gui.main import GUILogHandler, _setup_main_window, run_gui
from ert.gui.simulation.run_dialog import RunDialog
from ert.gui.tools.event_viewer import add_gui_log_handler
from ert.gui.tools.plot.plot_window import PlotWindow
from ert.services import StorageService


@pytest.mark.usefixtures("use_tmpdir")
def test_gui_load(qtbot):
    config_file = Path("config.ert")
    config_file.write_text("NUM_REALIZATIONS 1\n", encoding="utf-8")

    args = Mock()
    args.config = str(config_file)
    gui = _setup_main_window(
        EnKFMain(ErtConfig.from_file(str(config_file))), args, GUILogHandler()
    )
    qtbot.addWidget(gui)

    sim_panel = gui.findChild(QWidget, name="Simulation_panel")
    single_run_panel = gui.findChild(QWidget, name="Single_test_run_panel")
    assert (
        sim_panel.getCurrentSimulationModel() == single_run_panel.getSimulationModel()
    )

    sim_mode = gui.findChild(QWidget, name="Simulation_mode")
    qtbot.keyClick(sim_mode, Qt.Key_Down)

    ensemble_panel = gui.findChild(QWidget, name="Ensemble_experiment_panel")
    assert sim_panel.getCurrentSimulationModel() == ensemble_panel.getSimulationModel()


@pytest.mark.requires_window_manager
def test_gui_full(monkeypatch, tmp_path, qapp, mock_start_server, source_root):
    shutil.copytree(
        os.path.join(source_root, "test-data", "poly_example"),
        os.path.join(tmp_path, "poly_example"),
    )

    args = argparse.Namespace(config="poly_example/poly.ert", read_only=True)

    monkeypatch.chdir(tmp_path)

    qapp.exec_ = lambda: None  # exec_ starts the event loop, and will stall the test.
    monkeypatch.setattr(ert.gui.main, "QApplication", Mock(return_value=qapp))
    run_gui(args)
    mock_start_server.assert_called_once_with(
        project=str(tmp_path / "poly_example" / "storage"),
        ert_config="poly.ert",
    )


@pytest.mark.requires_window_manager
def test_that_loading_gui_creates_no_storage_in_read_only_mode(
    monkeypatch, tmp_path, qapp, source_root
):
    shutil.copytree(
        os.path.join(source_root, "test-data", "poly_example"),
        os.path.join(tmp_path, "poly_example"),
    )

    monkeypatch.chdir(tmp_path)

    args = argparse.Namespace(config="poly_example/poly.ert", read_only=True)

    qapp.exec_ = lambda: None  # exec_ starts the event loop, and will stall the test.
    monkeypatch.setattr(ert.gui.main, "QApplication", Mock(return_value=qapp))
    monkeypatch.setattr(ert.gui.main.LibresFacade, "enspath", tmp_path)
    run_gui(args)
    assert [p.stem for p in tmp_path.glob("**/*")].count("storage") == 0


@pytest.mark.usefixtures("use_tmpdir")
def test_gui_iter_num(monkeypatch, qtbot):
    config_file = Path("config.ert")
    config_file.write_text("NUM_REALIZATIONS 1\n", encoding="utf-8")

    args_mock = Mock()
    args_mock.config = str(config_file)

    # won't run simulations so we mock it and test whether "iter_num" is in arguments
    def _assert_iter_in_args(panel):
        assert panel.getSimulationArguments().iter_num == 10

    args_mock = Mock()
    args_mock.config = "poly.ert"
    type(args_mock).config = PropertyMock(return_value="config.ert")

    monkeypatch.setattr(
        ert.gui.simulation.simulation_panel.SimulationPanel,
        "runSimulation",
        _assert_iter_in_args,
    )

    gui = _setup_main_window(
        EnKFMain(ErtConfig.from_file(str(config_file))), args_mock, GUILogHandler()
    )
    qtbot.addWidget(gui)

    sim_mode = gui.findChild(QWidget, name="Simulation_mode")
    qtbot.keyClick(sim_mode, Qt.Key_Down)

    sim_panel = gui.findChild(QWidget, name="Simulation_panel")

    ensemble_panel = gui.findChild(QWidget, name="Ensemble_experiment_panel")
    # simulate entering number 10 as iter_num
    qtbot.keyClick(ensemble_panel._iter_field, Qt.Key_Backspace)
    qtbot.keyClicks(ensemble_panel._iter_field, "10")
    qtbot.keyClick(ensemble_panel._iter_field, Qt.Key_Enter)

    start_simulation = gui.findChild(QWidget, name="start_simulation")
    qtbot.mouseClick(start_simulation, Qt.LeftButton)
    assert sim_panel.getSimulationArguments().iter_num == 10


def test_that_gui_gives_suggestions_when_you_have_umask_in_config(
    monkeypatch, qapp, tmp_path
):
    config_file = tmp_path / "config.ert"
    config_file.write_text("NUM_REALIZATIONS 1\n UMASK 0222\n")

    args = Mock()
    args.config = str(config_file)
    with add_gui_log_handler() as log_handler:
        gui, _, _ = ert.gui.main._start_initial_gui_window(args, log_handler)
        assert gui.windowTitle() == "Some problems detected"


def test_that_errors_are_shown_in_the_suggester_window_when_present(
    monkeypatch, qapp, tmp_path
):
    config_file = tmp_path / "config.ert"
    config_file.write_text("NUM_REALIZATIONS you_cant_do_this\n")

    args = Mock()
    args.config = str(config_file)
    with add_gui_log_handler() as log_handler:
        gui, _, _ = ert.gui.main._start_initial_gui_window(args, log_handler)
        assert gui.windowTitle() == "Some problems detected"


@pytest.mark.usefixtures("copy_poly_case")
def test_that_the_ui_show_no_warnings_when_observations_found(qapp):
    args = Mock()
    args.config = "poly.ert"
    with add_gui_log_handler() as log_handler:
        gui, _, _ = ert.gui.main._start_initial_gui_window(args, log_handler)
        combo_box = gui.findChild(QComboBox, name="Simulation_mode")
        assert combo_box.count() == 5

        for i in range(combo_box.count()):
            assert combo_box.model().item(i).isEnabled()

        assert gui.windowTitle() == "ERT - poly.ert"


def test_that_the_ui_show_warnings_when_there_are_no_observations(qapp, tmp_path):
    config_file = tmp_path / "config.ert"
    config_file.write_text("NUM_REALIZATIONS 1\n")

    args = Mock()
    args.config = str(config_file)
    with add_gui_log_handler() as log_handler:
        gui, _, _ = ert.gui.main._start_initial_gui_window(args, log_handler)
        combo_box = gui.findChild(QComboBox, name="Simulation_mode")
        assert combo_box.count() == 5

        for i in range(2):
            assert combo_box.model().item(i).isEnabled()
        for i in range(2, 5):
            assert not combo_box.model().item(i).isEnabled()

        assert gui.windowTitle() == "ERT - config.ert"


@pytest.mark.usefixtures("copy_poly_case")
def test_that_ert_starts_when_there_are_no_problems(monkeypatch, qapp, tmp_path):
    args = Mock()
    args.config = "poly.ert"
    with add_gui_log_handler() as log_handler:
        gui, _, _ = ert.gui.main._start_initial_gui_window(args, log_handler)
        assert gui.windowTitle() == "ERT - poly.ert"


@pytest.mark.usefixtures("use_tmpdir")
def test_that_run_dialog_can_be_closed_after_used_to_open_plots(qtbot, storage):
    """
    This is a regression test for a bug where the plot window opened from run dialog
    would have run dialog as parent. Because of that it would be destroyed when
    run dialog was closed and end in a c++ QTObject lifetime crash.

    Also tests that the run_dialog is not modal (does not block the main_window),
    but simulations cannot be clicked from the main window while the run dialog is open.
    """
    config_file = Path("config.ert")
    config_file.write_text(
        f"NUM_REALIZATIONS 1\nENSPATH {storage.path}\n", encoding="utf-8"
    )

    args_mock = Mock()
    args_mock.config = str(config_file)

    ert_config = ErtConfig.from_file(str(config_file))
    enkf_main = EnKFMain(ert_config)
    with StorageService.init_service(
        ert_config=str(config_file),
        project=os.path.abspath(ert_config.ens_path),
    ):
        gui = _setup_main_window(enkf_main, args_mock, GUILogHandler())
        gui.notifier.set_storage(storage)
        qtbot.addWidget(gui)

        start_simulation = gui.findChild(QToolButton, name="start_simulation")
        assert isinstance(start_simulation, QToolButton)

        def handle_dialog():
            message_box = gui.findChild(QMessageBox)
            qtbot.mouseClick(message_box.button(QMessageBox.Yes), Qt.LeftButton)

        QTimer.singleShot(500, handle_dialog)

        def use_rundialog():
            qtbot.waitUntil(lambda: gui.findChild(RunDialog) is not None)

            # Ensure that once the run dialog is opened
            # another simulation cannot be started
            assert not start_simulation.isEnabled()

            run_dialog = gui.findChild(RunDialog)
            assert isinstance(run_dialog, RunDialog)

            # The user expects to be able to open e.g. the even viewer
            # while the run dialog is open
            assert not run_dialog.isModal()

            qtbot.mouseClick(run_dialog.plot_button, Qt.LeftButton)
            qtbot.waitUntil(run_dialog.done_button.isVisible, timeout=20000)
            qtbot.mouseClick(run_dialog.done_button, Qt.LeftButton)

            # Ensure that once the run dialog is closed
            # another simulation can be started
            assert start_simulation.isEnabled()

        QTimer.singleShot(1000, use_rundialog)
        qtbot.mouseClick(start_simulation, Qt.LeftButton)

        qtbot.waitUntil(lambda: gui.findChild(PlotWindow) is not None)

        # Cycle through showing all the tabs
        plot_window = gui.findChild(PlotWindow)
        for tab in plot_window._plot_widgets:
            plot_window._central_tab.setCurrentWidget(tab)


def test_help_buttons_in_suggester_dialog(monkeypatch, qapp, tmp_path, qtbot):
    """
    WHEN I am shown an error in the gui
    THEN the suggester gui comes up
    AND I can find the version of ert by opening the about panel
    AND go to github to submit an issue by clicking a button.
    """
    config_file = tmp_path / "config.ert"
    config_file.write_text(
        "NUM_REALIZATIONS 1\n RUNPATH iens-%d/iter-%d\n", encoding="utf-8"
    )

    args = Mock()
    args.config = str(config_file)
    with add_gui_log_handler() as log_handler:
        gui, _, _ = ert.gui.main._start_initial_gui_window(args, log_handler)
        assert gui.windowTitle() == "Some problems detected"

        about_button = gui.findChild(QWidget, name="about_button")
        qtbot.mouseClick(about_button, Qt.LeftButton)

        help_dialog = gui.findChild(AboutDialog)
        assert isinstance(help_dialog, AboutDialog)
        assert help_dialog.windowTitle() == "About"

        about_close_button = help_dialog.findChild(QWidget, name="close_button")
        qtbot.mouseClick(about_close_button, Qt.LeftButton)

        with patch("webbrowser.open", MagicMock(return_value=True)) as browser_open:
            github_button = gui.findChild(QWidget, name="GitHub page")
            qtbot.mouseClick(github_button, Qt.LeftButton)
            assert browser_open.called
