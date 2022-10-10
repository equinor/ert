import argparse
import os
import shutil
from unittest.mock import MagicMock, Mock, PropertyMock

import pytest
from qtpy.QtCore import Qt, QTimer
from qtpy.QtWidgets import QDialog, QMessageBox, QWidget

import ert.gui
from ert.gui.ertnotifier import ErtNotifier
from ert.gui.ertwidgets.message_box import ErtMessageBox
from ert.gui.gert_main import GUILogHandler, _start_window, run_gui


@pytest.fixture()
def patch_enkf_main(monkeypatch, tmp_path):
    plugins_mock = Mock()
    plugins_mock.getPluginJobs.return_value = []

    mocked_enkf_main = Mock()
    mocked_enkf_main.getWorkflowList.return_value = plugins_mock
    mocked_enkf_main.getEnsembleSize.return_value = 10
    mocked_enkf_main.storage_manager = []

    mocked_enkf_main.getWorkflowList.return_value.getWorkflowNames.return_value = [
        "my_workflow"
    ]

    res_config_mock = Mock()
    type(res_config_mock).config_path = PropertyMock(return_value=tmp_path)
    facade_mock = Mock()
    facade_mock.get_ensemble_size.return_value = 1
    facade_mock.get_number_of_iterations.return_value = 1
    monkeypatch.setattr(
        ert.gui.simulation.simulation_panel,
        "LibresFacade",
        Mock(return_value=facade_mock),
    )

    monkeypatch.setattr(
        ert.gui.simulation.ensemble_smoother_panel,
        "LibresFacade",
        Mock(return_value=facade_mock),
    )
    monkeypatch.setattr(
        ert.gui.ertwidgets.caseselector.CaseSelector,
        "_getAllCases",
        Mock(return_value=["test"]),
    )

    def patched_mask_to_rangestring(mask):
        return ""

    monkeypatch.setattr(
        "ert._c_wrappers.config.rangestring.mask_to_rangestring.__code__",
        patched_mask_to_rangestring.__code__,
    )

    monkeypatch.setattr(
        ert.gui.ertwidgets.summarypanel.ErtSummary,
        "getForwardModels",
        Mock(return_value=[]),
    )
    monkeypatch.setattr(
        ert.gui.ertwidgets.summarypanel.ErtSummary,
        "getParameters",
        Mock(return_value=[]),
    )
    monkeypatch.setattr(
        ert.gui.ertwidgets.summarypanel.ErtSummary,
        "getObservations",
        Mock(return_value=[]),
    )

    yield mocked_enkf_main


def test_gui_load(qtbot, patch_enkf_main):
    args = argparse.Namespace(config="does_not_matter.ert")
    notifier = ErtNotifier(args.config)
    gui = _start_window(patch_enkf_main, notifier, args, GUILogHandler())
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

    args = argparse.Namespace(config="poly_example/poly.ert")

    monkeypatch.chdir(tmp_path)

    qapp.exec_ = lambda: None  # exec_ starts the event loop, and will stall the test.
    monkeypatch.setattr(ert.gui.gert_main, "QApplication", Mock(return_value=qapp))
    monkeypatch.setattr(ert.gui.gert_main.LibresFacade, "enspath", tmp_path)
    run_gui(args)
    mock_start_server.assert_called_once_with(
        project=str(tmp_path), res_config="poly.ert"
    )


@pytest.mark.requires_window_manager
def test_that_loading_gui_creates_a_single_storage_folder(
    monkeypatch, tmp_path, qapp, source_root
):
    shutil.copytree(
        os.path.join(source_root, "test-data", "poly_example"),
        os.path.join(tmp_path, "poly_example"),
    )

    monkeypatch.chdir(tmp_path)

    args = argparse.Namespace(config="poly_example/poly.ert")

    qapp.exec_ = lambda: None  # exec_ starts the event loop, and will stall the test.
    monkeypatch.setattr(ert.gui.gert_main, "QApplication", Mock(return_value=qapp))
    monkeypatch.setattr(ert.gui.gert_main.LibresFacade, "enspath", tmp_path)
    run_gui(args)
    assert [p.stem for p in tmp_path.glob("**/*")].count("storage") == 1


def test_gui_iter_num(monkeypatch, qtbot, patch_enkf_main):
    # won't run simulations so we mock it and test whether "iter_num" is in arguments
    def _assert_iter_in_args(panel):
        assert panel.getSimulationArguments().iter_num == 10

    args_mock = Mock()
    type(args_mock).config = PropertyMock(return_value="config.ert")

    monkeypatch.setattr(
        ert.gui.simulation.simulation_panel.SimulationPanel,
        "runSimulation",
        _assert_iter_in_args,
    )

    notifier = ErtNotifier(args_mock.config)
    gui = _start_window(patch_enkf_main, notifier, args_mock, GUILogHandler())
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


def test_start_simulation_disabled(monkeypatch, qtbot, patch_enkf_main):
    args_mock = Mock()
    type(args_mock).config = PropertyMock(return_value="config.ert")

    dummy_run_dialog = QDialog(None)

    setattr(dummy_run_dialog, "startSimulation", lambda *args: None)

    monkeypatch.setattr(
        ert.gui.simulation.simulation_panel.QMessageBox,
        "question",
        lambda *args: QMessageBox.Yes,
    )
    monkeypatch.setattr(
        ert.gui.simulation.simulation_panel, "RunDialog", lambda *args: dummy_run_dialog
    )
    monkeypatch.setattr(
        ert.gui.simulation.simulation_panel, "create_model", lambda *args: None
    )

    notifier = MagicMock()
    gui = _start_window(patch_enkf_main, notifier, args_mock, GUILogHandler())
    qtbot.addWidget(gui)

    start_simulation = gui.findChild(QWidget, name="start_simulation")

    def handle_dialog():
        assert not start_simulation.isEnabled()
        dummy_run_dialog.accept()

    QTimer.singleShot(10, handle_dialog)
    qtbot.mouseClick(start_simulation, Qt.LeftButton)
    assert start_simulation.isEnabled()


def test_dialog(qtbot):
    msg = ErtMessageBox("Simulations failed!", "failed_msg\nwith two lines")
    qtbot.addWidget(msg)
    assert msg.label_text.text() == "Simulations failed!"
    assert msg.details_text.toPlainText() == "failed_msg\nwith two lines"
