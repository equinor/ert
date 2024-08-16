import contextlib
import copy
import fileinput
import os
import os.path
import shutil
import stat
import time
from contextlib import contextmanager
from datetime import datetime as dt
from pathlib import Path
from textwrap import dedent
from typing import Generator, List, Tuple, Type, TypeVar
from unittest.mock import MagicMock, Mock

import pytest
from pytestqt.qtbot import QtBot
from qtpy.QtCore import Qt, QTimer
from qtpy.QtWidgets import QApplication, QComboBox, QMessageBox, QPushButton, QWidget

from ert.config import ErtConfig
from ert.ensemble_evaluator.snapshot import (
    ForwardModel,
    RealizationSnapshot,
    Snapshot,
    SnapshotBuilder,
    SnapshotDict,
)
from ert.ensemble_evaluator.state import (
    ENSEMBLE_STATE_STARTED,
    FORWARD_MODEL_STATE_FINISHED,
    FORWARD_MODEL_STATE_START,
    REALIZATION_STATE_FAILED,
    REALIZATION_STATE_RUNNING,
    REALIZATION_STATE_UNKNOWN,
)
from ert.gui.ertwidgets import ClosableDialog
from ert.gui.ertwidgets.create_experiment_dialog import CreateExperimentDialog
from ert.gui.ertwidgets.ensembleselector import EnsembleSelector
from ert.gui.main import ErtMainWindow, _setup_main_window, add_gui_log_handler
from ert.gui.simulation.experiment_panel import ExperimentPanel
from ert.gui.simulation.run_dialog import RunDialog
from ert.gui.simulation.view import RealizationWidget
from ert.gui.tools.load_results.load_results_panel import LoadResultsPanel
from ert.gui.tools.manage_experiments.manage_experiments_tool import (
    ManageExperimentsTool,
)
from ert.gui.tools.manage_experiments.storage_widget import AddWidget, StorageWidget
from ert.plugins import ErtPluginContext
from ert.run_models import EnsembleExperiment, MultipleDataAssimilation
from ert.services import StorageService
from ert.storage import Storage, open_storage
from tests.unit_tests.gui.simulation.test_run_path_dialog import handle_run_path_dialog


@pytest.fixture
def opened_main_window(
    source_root, tmp_path, monkeypatch
) -> Generator[ErtMainWindow, None, None]:
    monkeypatch.chdir(tmp_path)
    _new_poly_example(source_root, tmp_path)
    with _open_main_window(tmp_path / "poly.ert") as (
        gui,
        storage,
        config,
    ), StorageService.init_service(
        project=os.path.abspath(config.ens_path),
    ):
        _add_default_ensemble(storage, gui, config)
        yield gui


def _new_poly_example(source_root, destination):
    shutil.copytree(
        os.path.join(source_root, "test-data", "poly_example"),
        destination,
        dirs_exist_ok=True,
    )

    with fileinput.input(destination / "poly.ert", inplace=True) as fin:
        for line in fin:
            if "NUM_REALIZATIONS" in line:
                # Decrease the number of realizations to speed up the test,
                # if there is flakyness, this can be increased.
                print("NUM_REALIZATIONS 20", end="\n")
            else:
                print(line, end="")


def _add_default_ensemble(storage: Storage, gui: ErtMainWindow, config: ErtConfig):
    gui.notifier.set_current_ensemble(
        storage.create_experiment(
            parameters=config.ensemble_config.parameter_configuration,
            observations=config.observations,
        ).create_ensemble(
            name="default",
            ensemble_size=config.model_config.num_realizations,
        )
    )


@contextmanager
def _open_main_window(
    path,
) -> Generator[Tuple[ErtMainWindow, Storage, ErtConfig], None, None]:
    args_mock = Mock()
    args_mock.config = str(path)
    with ErtPluginContext():
        config = ErtConfig.with_plugins().from_file(path)
        with open_storage(
            config.ens_path, mode="w"
        ) as storage, add_gui_log_handler() as log_handler:
            gui = _setup_main_window(config, args_mock, log_handler, storage)
            yield gui, storage, config
            gui.close()


@pytest.fixture
def opened_main_window_clean(source_root, tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    _new_poly_example(source_root, tmp_path)
    with _open_main_window(tmp_path / "poly.ert") as (
        gui,
        _,
        config,
    ), StorageService.init_service(
        project=os.path.abspath(config.ens_path),
    ):
        yield gui


@pytest.fixture
def opened_main_window_snake_oil(snake_oil_case_storage):
    with _open_main_window(Path("./snake_oil.ert")) as (
        gui,
        _,
        config,
    ), StorageService.init_service(
        project=os.path.abspath(config.ens_path),
    ):
        yield gui


@pytest.fixture(scope="module")
def _esmda_run(run_experiment, source_root, tmp_path_factory):
    path = tmp_path_factory.mktemp("test-data")
    _new_poly_example(source_root, path)
    with pytest.MonkeyPatch.context() as mp, _open_main_window(path / "poly.ert") as (
        gui,
        storage,
        config,
    ):
        mp.chdir(path)
        _add_default_ensemble(storage, gui, config)
        run_experiment(MultipleDataAssimilation, gui)
        # Check that we produce update log
        log_paths = list(Path(config.analysis_config.log_path).iterdir())
        assert log_paths
        assert (log_paths[0] / "Report.report").exists()
        assert (log_paths[0] / "Report.csv").exists()

    return path


def _ensemble_experiment_run(
    run_experiment, source_root, tmp_path_factory, failing_reals
):
    path = tmp_path_factory.mktemp("test-data")
    _new_poly_example(source_root, path)
    with pytest.MonkeyPatch.context() as mp, _open_main_window(path / "poly.ert") as (
        gui,
        storage,
        config,
    ):
        mp.chdir(path)
        if failing_reals:
            with open("poly_eval.py", "w", encoding="utf-8") as f:
                f.write(
                    dedent(
                        """\
                        #!/usr/bin/env python3
                        import os
                        import sys
                        import json

                        def _load_coeffs(filename):
                            with open(filename, encoding="utf-8") as f:
                                return json.load(f)["COEFFS"]

                        def _evaluate(coeffs, x):
                            return coeffs["a"] * x**2 + coeffs["b"] * x + coeffs["c"]

                        if __name__ == "__main__":
                            if int(os.getenv("_ERT_REALIZATION_NUMBER")) % 2 == 0:
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
        _add_default_ensemble(storage, gui, config)
        run_experiment(EnsembleExperiment, gui)

    return path


@pytest.fixture
def esmda_has_run(_esmda_run, tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    shutil.copytree(_esmda_run, tmp_path, dirs_exist_ok=True)
    with _open_main_window(tmp_path / "poly.ert") as (
        gui,
        _,
        config,
    ), StorageService.init_service(
        project=os.path.abspath(config.ens_path),
    ):
        yield gui


@pytest.fixture
def ensemble_experiment_has_run(
    tmp_path, monkeypatch, run_experiment, source_root, tmp_path_factory
):
    monkeypatch.chdir(tmp_path)
    test_files = _ensemble_experiment_run(
        run_experiment, source_root, tmp_path_factory, True
    )
    shutil.copytree(test_files, tmp_path, dirs_exist_ok=True)
    with _open_main_window(tmp_path / "poly.ert") as (
        gui,
        _,
        config,
    ), StorageService.init_service(
        project=os.path.abspath(config.ens_path),
    ):
        yield gui


@pytest.fixture
def ensemble_experiment_has_run_no_failure(
    tmp_path, monkeypatch, run_experiment, source_root, tmp_path_factory
):
    monkeypatch.chdir(tmp_path)
    test_files = _ensemble_experiment_run(
        run_experiment, source_root, tmp_path_factory, False
    )
    shutil.copytree(test_files, tmp_path, dirs_exist_ok=True)
    with _open_main_window(tmp_path / "poly.ert") as (
        gui,
        _,
        config,
    ), StorageService.init_service(
        project=os.path.abspath(config.ens_path),
    ):
        yield gui


@pytest.fixture(name="run_experiment", scope="module")
def run_experiment_fixture(request):
    def func(experiment_mode, gui):
        qtbot = QtBot(request)
        with contextlib.suppress(FileNotFoundError):
            shutil.rmtree("poly_out")
        # Select correct experiment in the simulation panel
        experiment_panel = gui.findChild(ExperimentPanel)
        assert isinstance(experiment_panel, ExperimentPanel)
        simulation_mode_combo = experiment_panel.findChild(QComboBox)
        assert isinstance(simulation_mode_combo, QComboBox)
        simulation_mode_combo.setCurrentText(experiment_mode.name())
        simulation_settings = experiment_panel._experiment_widgets[
            experiment_panel.get_current_experiment_type()
        ]
        if hasattr(simulation_settings, "_ensemble_name_field"):
            simulation_settings._ensemble_name_field.setText("iter-0")

        # Click start simulation and agree to the message
        run_experiment = experiment_panel.findChild(QWidget, name="run_experiment")

        def handle_dialog():
            QTimer.singleShot(
                500,
                lambda: handle_run_path_dialog(gui, qtbot, delete_run_path=False),
            )

        if not experiment_mode.name() in (
            "Ensemble experiment",
            "Evaluate ensemble",
        ):
            QTimer.singleShot(500, handle_dialog)
        qtbot.mouseClick(run_experiment, Qt.LeftButton)

        # The Run dialog opens, click show details and wait until done appears
        # then click it
        qtbot.waitUntil(lambda: gui.findChild(RunDialog) is not None)
        run_dialog = gui.findChild(RunDialog)

        qtbot.waitUntil(run_dialog.done_button.isVisible, timeout=200000)
        qtbot.waitUntil(lambda: run_dialog._tab_widget.currentWidget() is not None)

        # Assert that the number of boxes in the detailed view is
        # equal to the number of realizations
        realization_widget = run_dialog._tab_widget.currentWidget()
        assert isinstance(realization_widget, RealizationWidget)
        list_model = realization_widget._real_view.model()
        assert (
            list_model.rowCount()
            == experiment_panel.config.model_config.num_realizations
        )

        qtbot.mouseClick(run_dialog.done_button, Qt.LeftButton)

    return func


@pytest.fixture()
def full_snapshot() -> Snapshot:
    real = RealizationSnapshot(
        status=REALIZATION_STATE_RUNNING,
        active=True,
        forward_models={
            "0": ForwardModel(
                start_time=dt.now(),
                end_time=dt.now(),
                name="poly_eval",
                index="0",
                status=FORWARD_MODEL_STATE_START,
                error="error",
                stdout="std_out_file",
                stderr="std_err_file",
                current_memory_usage="123",
                max_memory_usage="312",
            ),
            "1": ForwardModel(
                start_time=dt.now(),
                end_time=dt.now(),
                name="poly_postval",
                index="1",
                status=FORWARD_MODEL_STATE_START,
                error="error",
                stdout="std_out_file",
                stderr="std_err_file",
                current_memory_usage="123",
                max_memory_usage="312",
            ),
            "2": ForwardModel(
                start_time=dt.now(),
                end_time=None,
                name="poly_post_mortem",
                index="2",
                status=FORWARD_MODEL_STATE_START,
                error="error",
                stdout="std_out_file",
                stderr="std_err_file",
                current_memory_usage="123",
                max_memory_usage="312",
            ),
            "3": ForwardModel(
                start_time=dt.now(),
                end_time=None,
                name="poly_not_started",
                index="3",
                status=FORWARD_MODEL_STATE_START,
                error="error",
                stdout="std_out_file",
                stderr="std_err_file",
                current_memory_usage="123",
                max_memory_usage="312",
            ),
        },
    )
    snapshot = SnapshotDict(
        status=ENSEMBLE_STATE_STARTED,
        reals={},
    )
    for i in range(0, 100):
        snapshot.reals[str(i)] = copy.deepcopy(real)

    return Snapshot.from_nested_dict(snapshot.model_dump())


@pytest.fixture()
def fail_snapshot() -> Snapshot:
    real = RealizationSnapshot(
        status=REALIZATION_STATE_FAILED,
        active=True,
        forward_models={
            "0": ForwardModel(
                start_time=dt.now(),
                end_time=dt.now(),
                name="poly_eval",
                index="0",
                status=FORWARD_MODEL_STATE_FINISHED,
                error="error",
                stdout="std_out_file",
                stderr="std_err_file",
                current_memory_usage="123",
                max_memory_usage="312",
            )
        },
    )
    snapshot = SnapshotDict(
        status=ENSEMBLE_STATE_STARTED,
        reals={},
    )
    for i in range(0, 1):
        snapshot.reals[str(i)] = copy.deepcopy(real)

    return Snapshot.from_nested_dict(snapshot.model_dump())


@pytest.fixture()
def large_snapshot() -> Snapshot:
    builder = SnapshotBuilder()
    for i in range(0, 150):
        builder.add_forward_model(
            forward_model_id=str(i),
            index=str(i),
            name=f"job_{i}",
            current_memory_usage="500",
            max_memory_usage="1000",
            status=FORWARD_MODEL_STATE_START,
            stdout=f"job_{i}.stdout",
            stderr=f"job_{i}.stderr",
            start_time=dt(1999, 1, 1),
            end_time=dt(2019, 1, 1),
        )
    real_ids = [str(i) for i in range(0, 150)]
    return builder.build(real_ids, REALIZATION_STATE_UNKNOWN)


@pytest.fixture()
def small_snapshot() -> Snapshot:
    builder = SnapshotBuilder()
    for i in range(0, 2):
        builder.add_forward_model(
            forward_model_id=str(i),
            index=str(i),
            name=f"job_{i}",
            current_memory_usage="500",
            max_memory_usage="1000",
            status=FORWARD_MODEL_STATE_START,
            stdout=f"job_{i}.stdout",
            stderr=f"job_{i}.stderr",
            start_time=dt(1999, 1, 1),
            end_time=dt(2019, 1, 1),
        )
    real_ids = [str(i) for i in range(0, 5)]
    return builder.build(real_ids, REALIZATION_STATE_UNKNOWN)


@pytest.fixture(name="active_realizations")
def active_realizations_fixture() -> Mock:
    active_reals = MagicMock()
    active_reals.count = Mock(return_value=10)
    active_reals.__iter__.return_value = [True] * 10
    return active_reals


@pytest.fixture
def runmodel(active_realizations) -> Mock:
    brm = Mock()
    brm.get_runtime = Mock(return_value=100)
    brm.format_error = Mock(return_value="")
    brm.support_restart = True
    brm.simulation_arguments = {"active_realizations": active_realizations}
    brm.has_failed_realizations = lambda: False
    return brm


class MockTracker:
    def __init__(self, events) -> None:
        self._events = events
        self._is_running = True

    def track(self):
        for event in self._events:
            if not self._is_running:
                break
            time.sleep(0.1)
            yield event

    def reset(self):
        pass

    def request_termination(self):
        self._is_running = False


@pytest.fixture
def mock_tracker():
    def _make_mock_tracker(events):
        return MockTracker(events)

    return _make_mock_tracker


def load_results_manually(qtbot, gui, ensemble_name="default"):
    def handle_load_results_dialog():
        dialog = wait_for_child(gui, qtbot, ClosableDialog)
        panel = get_child(dialog, LoadResultsPanel)

        ensemble_selector = get_child(panel, EnsembleSelector)
        index = ensemble_selector.findText(ensemble_name, Qt.MatchFlag.MatchContains)
        assert index != -1
        ensemble_selector.setCurrentIndex(index)

        # click on "Load"
        load_button = get_child(panel.parent(), QPushButton, name="Load")

        # Verify that the messagebox is the success kind
        def handle_popup_dialog():
            messagebox = QApplication.activeModalWidget()
            assert isinstance(messagebox, QMessageBox)
            assert messagebox.text() == "Successfully loaded all realisations"
            ok_button = messagebox.button(QMessageBox.Ok)
            qtbot.mouseClick(ok_button, Qt.LeftButton)

        QTimer.singleShot(2000, handle_popup_dialog)
        qtbot.mouseClick(load_button, Qt.LeftButton)
        dialog.close()

    QTimer.singleShot(1000, handle_load_results_dialog)
    load_results_tool = gui.tools["Load results manually"]
    load_results_tool.trigger()


def add_experiment_manually(
    qtbot, gui, experiment_name="My experiment", ensemble_name="default"
):
    manage_tool = gui.tools["Manage experiments"]
    manage_tool.trigger()

    assert isinstance(manage_tool, ManageExperimentsTool)
    experiments_panel = manage_tool._manage_experiments_panel

    # Open the create new experiment tab
    experiments_panel.setCurrentIndex(0)
    current_tab = experiments_panel.currentWidget()
    assert current_tab.objectName() == "create_new_ensemble_tab"
    storage_widget = get_child(current_tab, StorageWidget)

    def handle_add_dialog():
        dialog = wait_for_child(current_tab, qtbot, CreateExperimentDialog)
        dialog._experiment_edit.setText(experiment_name)
        dialog._ensemble_edit.setText(ensemble_name)
        qtbot.mouseClick(dialog._ok_button, Qt.MouseButton.LeftButton)

    QTimer.singleShot(1000, handle_add_dialog)
    add_widget = get_child(storage_widget, AddWidget)
    qtbot.mouseClick(add_widget.addButton, Qt.MouseButton.LeftButton)

    experiments_panel.close()


V = TypeVar("V")


def wait_for_child(gui, qtbot: QtBot, typ: Type[V], *args, **kwargs) -> V:
    qtbot.waitUntil(lambda: gui.findChild(typ, *args, **kwargs) is not None)
    return get_child(gui, typ, *args, **kwargs)


def get_child(gui: QWidget, typ: Type[V], *args, **kwargs) -> V:
    child = gui.findChild(typ, *args, **kwargs)
    assert isinstance(child, typ)
    return child


def get_children(gui: QWidget, typ: Type[V], *args, **kwargs) -> List[V]:
    children: List[typ] = gui.findChildren(typ, *args, **kwargs)
    return children
