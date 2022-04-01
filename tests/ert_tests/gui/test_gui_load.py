import os
from distutils.version import StrictVersion
from unittest.mock import Mock, PropertyMock

import pytest
import qtpy
from qtpy.QtCore import Qt

import ert_gui
from ert_gui.ertnotifier import ErtNotifier
from ert_gui.gert_main import _start_window, run_gui


@pytest.fixture()
def patch_enkf_main(monkeypatch, tmpdir):
    plugins_mock = Mock()
    plugins_mock.getPluginJobs.return_value = []

    mocked_enkf_main = Mock()
    mocked_enkf_main.getWorkflowList.return_value = plugins_mock
    mocked_enkf_main.getEnsembleSize.return_value = 10

    mocked_enkf_main.getWorkflowList.return_value.getWorkflowNames.return_value = [
        "my_workflow"
    ]

    res_config_mock = Mock()
    type(res_config_mock).config_path = PropertyMock(return_value=tmpdir.strpath)
    facade_mock = Mock()
    facade_mock.get_analysis_module_names.return_value = []
    facade_mock.get_ensemble_size.return_value = 1
    facade_mock.get_number_of_iterations.return_value = 1
    monkeypatch.setattr(
        ert_gui.simulation.simulation_panel,
        "LibresFacade",
        Mock(return_value=facade_mock),
    )

    monkeypatch.setattr(
        ert_gui.simulation.ensemble_smoother_panel,
        "LibresFacade",
        Mock(return_value=facade_mock),
    )
    monkeypatch.setattr(
        ert_gui.gert_main, "EnKFMain", Mock(return_value=mocked_enkf_main)
    )
    monkeypatch.setattr(
        ert_gui.gert_main, "ResConfig", Mock(return_value=res_config_mock)
    )
    monkeypatch.setattr(
        ert_gui.ertwidgets.caseselector.CaseSelector,
        "_getAllCases",
        Mock(return_value=["test"]),
    )

    monkeypatch.setattr(
        ert_gui.ertwidgets.models.activerealizationsmodel,
        "mask_to_rangestring",
        Mock(return_value=""),
    )

    monkeypatch.setattr(
        ert_gui.ertwidgets.summarypanel.ErtSummary,
        "getForwardModels",
        Mock(return_value=[]),
    )
    monkeypatch.setattr(
        ert_gui.ertwidgets.summarypanel.ErtSummary,
        "getParameters",
        Mock(return_value=[]),
    )
    monkeypatch.setattr(
        ert_gui.ertwidgets.summarypanel.ErtSummary,
        "getObservations",
        Mock(return_value=[]),
    )

    yield mocked_enkf_main


@pytest.mark.skipif(
    StrictVersion(qtpy.PYQT_VERSION) < StrictVersion("5.0")
    and os.environ.get("PYTEST_QT_API") != "pyqt4v2",
    reason="PyQt4 with PYTEST_QT_API env. variable != pyqt4v2",
)
@pytest.mark.skipif(
    os.environ.get("TRAVIS_OS_NAME") == "osx", reason="xvfb not available on travis OSX"
)
def test_gui_load(monkeypatch, tmpdir, qtbot, patch_enkf_main):
    args_mock = Mock()
    type(args_mock).config = PropertyMock(return_value="config.ert")
    notifier = ErtNotifier(args_mock.config)
    gui = _start_window(patch_enkf_main, notifier, args_mock)
    qtbot.addWidget(gui)

    sim_panel = gui.findChild(qtpy.QtWidgets.QWidget, name="Simulation_panel")
    single_run_panel = gui.findChild(
        qtpy.QtWidgets.QWidget, name="Single_test_run_panel"
    )
    assert (
        sim_panel.getCurrentSimulationModel() == single_run_panel.getSimulationModel()
    )

    sim_mode = gui.findChild(qtpy.QtWidgets.QWidget, name="Simulation_mode")
    qtbot.keyClick(sim_mode, Qt.Key_Down)

    ensemble_panel = gui.findChild(
        qtpy.QtWidgets.QWidget, name="Ensemble_experiment_panel"
    )
    assert sim_panel.getCurrentSimulationModel() == ensemble_panel.getSimulationModel()


@pytest.mark.skipif(
    os.environ.get("TRAVIS_OS_NAME") == "osx", reason="xvfb not available on travis OSX"
)
@pytest.mark.skipif(
    StrictVersion(qtpy.PYQT_VERSION) < StrictVersion("5.0")
    and os.environ.get("PYTEST_QT_API") != "pyqt4v2",
    reason="PyQt4 with PYTEST_QT_API env. variable != pyqt4v2",
)
@pytest.mark.usefixtures("patch_enkf_main")
def test_gui_full(monkeypatch, tmpdir, qapp, mock_start_server):
    with tmpdir.as_cwd():
        args_mock = Mock()
        type(args_mock).config = PropertyMock(return_value="config.ert")

        qapp.exec_ = (
            lambda: None
        )  # exec_ starts the event loop, and will stall the test.
        monkeypatch.setattr(ert_gui.gert_main, "QApplication", Mock(return_value=qapp))
        gui = run_gui(args_mock)
        mock_start_server.assert_called_once_with(res_config="config.ert")


def test_gui_iter_num(monkeypatch, tmpdir, qtbot, patch_enkf_main):
    # won't run simulations so we mock it and test whether "iter_num" is in arguments
    def _assert_iter_in_args(panel):
        assert "iter_num" in panel.getSimulationArguments()

    args_mock = Mock()
    type(args_mock).config = PropertyMock(return_value="config.ert")

    monkeypatch.setattr(
        ert_gui.simulation.simulation_panel.SimulationPanel,
        "runSimulation",
        _assert_iter_in_args,
    )

    notifier = ErtNotifier(args_mock.config)
    gui = _start_window(patch_enkf_main, notifier, args_mock)
    qtbot.addWidget(gui)

    sim_mode = gui.findChild(qtpy.QtWidgets.QWidget, name="Simulation_mode")
    qtbot.keyClick(sim_mode, Qt.Key_Down)

    sim_panel = gui.findChild(qtpy.QtWidgets.QWidget, name="Simulation_panel")

    ensemble_panel = gui.findChild(
        qtpy.QtWidgets.QWidget, name="Ensemble_experiment_panel"
    )
    # simulate entering number 10 as iter_num
    qtbot.keyClick(ensemble_panel._iter_field, Qt.Key_Backspace)
    qtbot.keyClicks(ensemble_panel._iter_field, "10")
    qtbot.keyClick(ensemble_panel._iter_field, Qt.Key_Enter)

    start_simulation = gui.findChild(qtpy.QtWidgets.QWidget, name="start_simulation")
    qtbot.mouseClick(start_simulation, Qt.LeftButton)
    assert sim_panel.getSimulationArguments()["iter_num"] == 10
