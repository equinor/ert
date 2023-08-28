import contextlib
import copy
import fileinput
import os.path
import shutil
import stat
import time
from datetime import datetime as dt
from textwrap import dedent
from typing import Tuple
from unittest.mock import MagicMock, Mock

import pytest
from pytestqt.qtbot import QtBot
from qtpy.QtCore import Qt, QTimer
from qtpy.QtWidgets import QComboBox, QMessageBox, QWidget

from ert.config import ErtConfig
from ert.enkf_main import EnKFMain
from ert.ensemble_evaluator.snapshot import (
    Job,
    RealizationSnapshot,
    Snapshot,
    SnapshotBuilder,
    SnapshotDict,
    Step,
)
from ert.ensemble_evaluator.state import (
    ENSEMBLE_STATE_STARTED,
    JOB_STATE_START,
    REALIZATION_STATE_UNKNOWN,
    STEP_STATE_UNKNOWN,
)
from ert.gui.ertwidgets.caselist import AddRemoveWidget, CaseList
from ert.gui.ertwidgets.closabledialog import ClosableDialog
from ert.gui.ertwidgets.validateddialog import ValidatedDialog
from ert.gui.main import GUILogHandler, _setup_main_window
from ert.gui.simulation.run_dialog import RunDialog
from ert.gui.simulation.simulation_panel import SimulationPanel
from ert.gui.simulation.view import RealizationWidget
from ert.gui.tools.manage_cases.case_init_configuration import (
    CaseInitializationConfigurationPanel,
)
from ert.run_models import EnsembleExperiment, MultipleDataAssimilation
from ert.services import StorageService
from ert.storage import open_storage


def find_cases_dialog_and_panel(
    gui, qtbot: QtBot
) -> Tuple[ClosableDialog, CaseInitializationConfigurationPanel]:
    qtbot.waitUntil(lambda: gui.findChild(ClosableDialog, "manage-cases") is not None)
    dialog = gui.findChild(ClosableDialog, "manage-cases")
    cases_panel = dialog.findChild(CaseInitializationConfigurationPanel)
    assert isinstance(cases_panel, CaseInitializationConfigurationPanel)
    return (dialog, cases_panel)


@pytest.mark.usefixtures("use_tmpdir")
@pytest.fixture(name="opened_main_window", scope="module")
def opened_main_window_fixture(source_root, tmpdir_factory):
    with pytest.MonkeyPatch.context() as mp:
        tmp_path = tmpdir_factory.mktemp("test-data")
        shutil.copytree(
            os.path.join(source_root, "test-data", "poly_example"),
            tmp_path / "test_data",
        )
        mp.chdir(tmp_path / "test_data")
        with fileinput.input("poly.ert", inplace=True) as fin:
            for line in fin:
                if "NUM_REALIZATIONS" in line:
                    # Decrease the number of realizations to speed up the test,
                    # if there is flakyness, this can be increased.
                    print("NUM_REALIZATIONS 20", end="\n")
                else:
                    print(line, end="")
            poly_case = EnKFMain(ErtConfig.from_file("poly.ert"))
        args_mock = Mock()
        args_mock.config = "poly.ert"

        with StorageService.init_service(
            ert_config=args_mock.config,
            project=os.path.abspath(poly_case.ert_config.ens_path),
        ), open_storage(poly_case.ert_config.ens_path, mode="w") as storage:
            gui = _setup_main_window(poly_case, args_mock, GUILogHandler())
            gui.notifier.set_storage(storage)
            gui.notifier.set_current_case(
                storage.create_experiment(
                    parameters=poly_case.ensembleConfig().parameter_configuration
                ).create_ensemble(
                    name="default",
                    ensemble_size=poly_case.getEnsembleSize(),
                )
            )
            yield gui
            gui.close()


@pytest.fixture
def opened_main_window_clean(source_root, tmpdir):
    with pytest.MonkeyPatch.context() as mp:
        shutil.copytree(
            os.path.join(source_root, "test-data", "poly_example"),
            tmpdir / "test_data",
        )
        mp.chdir(tmpdir / "test_data")

        with fileinput.input("poly.ert", inplace=True) as fin:
            for line in fin:
                if "NUM_REALIZATIONS" in line:
                    # Decrease the number of realizations to speed up the test,
                    # if there is flakyness, this can be increased.
                    print("NUM_REALIZATIONS 20", end="\n")
                else:
                    print(line, end="")

        poly_case = EnKFMain(ErtConfig.from_file("poly.ert"))
        args_mock = Mock()
        args_mock.config = "poly.ert"

        with StorageService.init_service(
            ert_config=args_mock.config,
            project=os.path.abspath(poly_case.ert_config.ens_path),
        ), open_storage(poly_case.ert_config.ens_path, mode="w") as storage:
            gui = _setup_main_window(poly_case, args_mock, GUILogHandler())
            gui.notifier.set_storage(storage)
            yield gui


@pytest.mark.usefixtures("use_tmpdir, opened_main_window")
@pytest.fixture(scope="module")
def esmda_has_run(run_experiment):
    # Runs a default ES-MDA run
    run_experiment(MultipleDataAssimilation)


@pytest.mark.usefixtures("use_tmpdir")
@pytest.fixture(name="run_experiment", scope="module")
def run_experiment_fixture(request, opened_main_window):
    def func(experiment_mode):
        qtbot = QtBot(request)
        gui = opened_main_window
        with contextlib.suppress(FileNotFoundError):
            shutil.rmtree("poly_out")
        # Select correct experiment in the simulation panel
        simulation_panel = gui.findChild(SimulationPanel)
        assert isinstance(simulation_panel, SimulationPanel)
        simulation_mode_combo = simulation_panel.findChild(QComboBox)
        assert isinstance(simulation_mode_combo, QComboBox)
        simulation_mode_combo.setCurrentText(experiment_mode.name())

        # Click start simulation and agree to the message
        start_simulation = simulation_panel.findChild(QWidget, name="start_simulation")

        def handle_dialog():
            message_box = gui.findChild(QMessageBox)
            qtbot.mouseClick(message_box.buttons()[0], Qt.LeftButton)

        QTimer.singleShot(500, handle_dialog)
        qtbot.mouseClick(start_simulation, Qt.LeftButton)

        # The Run dialog opens, click show details and wait until done appears
        # then click it
        qtbot.waitUntil(lambda: gui.findChild(RunDialog) is not None)
        run_dialog = gui.findChild(RunDialog)

        qtbot.mouseClick(run_dialog.show_details_button, Qt.LeftButton)

        qtbot.waitUntil(run_dialog.done_button.isVisible, timeout=200000)
        qtbot.waitUntil(lambda: run_dialog._tab_widget.currentWidget() is not None)

        # Assert that the number of boxes in the detailed view is
        # equal to the number of realizations
        realization_widget = run_dialog._tab_widget.currentWidget()
        assert isinstance(realization_widget, RealizationWidget)
        list_model = realization_widget._real_view.model()
        assert list_model.rowCount() == simulation_panel.ert.getEnsembleSize()

        qtbot.mouseClick(run_dialog.done_button, Qt.LeftButton)

    return func


@pytest.mark.usefixtures("use_tmpdir")
@pytest.fixture(scope="module")
def ensemble_experiment_has_run(opened_main_window, run_experiment, request):
    gui = opened_main_window
    qtbot = QtBot(request)

    def handle_dialog():
        dialog, cases_panel = find_cases_dialog_and_panel(gui, qtbot)

        # Open the create new cases tab
        cases_panel.setCurrentIndex(0)
        current_tab = cases_panel.currentWidget()
        assert current_tab.objectName() == "create_new_case_tab"
        create_widget = current_tab.findChild(AddRemoveWidget)
        case_list = current_tab.findChild(CaseList)
        assert isinstance(case_list, CaseList)

        # Click add case and name it "iter-0"
        def handle_add_dialog():
            qtbot.waitUntil(lambda: current_tab.findChild(ValidatedDialog) is not None)
            dialog = gui.findChild(ValidatedDialog)
            dialog.param_name.setText("iter-0")
            qtbot.mouseClick(dialog.ok_button, Qt.LeftButton)

        QTimer.singleShot(1000, handle_add_dialog)
        qtbot.mouseClick(create_widget.addButton, Qt.LeftButton)

        dialog.close()

    QTimer.singleShot(1000, handle_dialog)
    manage_tool = gui.tools["Manage cases"]
    manage_tool.trigger()

    with open("poly_eval.py", "w", encoding="utf-8") as f:
        f.write(
            dedent(
                """\
                #!/usr/bin/env python
                import numpy as np
                import sys
                import json

                def _load_coeffs(filename):
                    with open(filename, encoding="utf-8") as f:
                        return json.load(f)

                def _evaluate(coeffs, x):
                    return coeffs["a"] * x**2 + coeffs["b"] * x + coeffs["c"]

                if __name__ == "__main__":
                    if np.random.random(1) > 0.5:
                        sys.exit(1)
                    coeffs = _load_coeffs("coeffs.json")
                    output = [_evaluate(coeffs, x) for x in range(10)]
                    with open("poly.out", "w", encoding="utf-8") as f:
                        f.write("\\n".join(map(str, output)))
                """
            )
        )
    os.chmod(
        "poly_eval.py",
        os.stat("poly_eval.py").st_mode | stat.S_IXUSR | stat.S_IXGRP | stat.S_IXOTH,
    )
    run_experiment(EnsembleExperiment)


@pytest.fixture()
def full_snapshot() -> Snapshot:
    real = RealizationSnapshot(
        status=REALIZATION_STATE_UNKNOWN,
        active=True,
        steps={
            "0": Step(
                status="",
                jobs={
                    "0": Job(
                        start_time=dt.now(),
                        end_time=dt.now(),
                        name="poly_eval",
                        index="0",
                        status=JOB_STATE_START,
                        error="error",
                        stdout="std_out_file",
                        stderr="std_err_file",
                        current_memory_usage="123",
                        max_memory_usage="312",
                    ),
                    "1": Job(
                        start_time=dt.now(),
                        end_time=dt.now(),
                        name="poly_postval",
                        index="1",
                        status=JOB_STATE_START,
                        error="error",
                        stdout="std_out_file",
                        stderr="std_err_file",
                        current_memory_usage="123",
                        max_memory_usage="312",
                    ),
                    "2": Job(
                        start_time=dt.now(),
                        end_time=None,
                        name="poly_post_mortem",
                        index="2",
                        status=JOB_STATE_START,
                        error="error",
                        stdout="std_out_file",
                        stderr="std_err_file",
                        current_memory_usage="123",
                        max_memory_usage="312",
                    ),
                },
            )
        },
    )
    snapshot = SnapshotDict(
        status=ENSEMBLE_STATE_STARTED,
        reals={},
    )
    for i in range(0, 100):
        snapshot.reals[str(i)] = copy.deepcopy(real)

    return Snapshot(snapshot.dict())


@pytest.fixture()
def large_snapshot() -> Snapshot:
    builder = SnapshotBuilder().add_step(step_id="0", status=STEP_STATE_UNKNOWN)
    for i in range(0, 150):
        builder.add_job(
            step_id="0",
            job_id=str(i),
            index=str(i),
            name=f"job_{i}",
            current_memory_usage="500",
            max_memory_usage="1000",
            status=JOB_STATE_START,
            stdout=f"job_{i}.stdout",
            stderr=f"job_{i}.stderr",
            start_time=dt(1999, 1, 1).isoformat(),
            end_time=dt(2019, 1, 1).isoformat(),
        )
    real_ids = [str(i) for i in range(0, 150)]
    return builder.build(real_ids, REALIZATION_STATE_UNKNOWN)


@pytest.fixture()
def small_snapshot() -> Snapshot:
    builder = SnapshotBuilder().add_step(step_id="0", status=STEP_STATE_UNKNOWN)
    for i in range(0, 2):
        builder.add_job(
            step_id="0",
            job_id=str(i),
            index=str(i),
            name=f"job_{i}",
            current_memory_usage="500",
            max_memory_usage="1000",
            status=JOB_STATE_START,
            stdout=f"job_{i}.stdout",
            stderr=f"job_{i}.stderr",
            start_time=dt(1999, 1, 1).isoformat(),
            end_time=dt(2019, 1, 1).isoformat(),
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
    brm.hasRunFailed = Mock(return_value=False)
    brm.getFailMessage = Mock(return_value="")
    brm.support_restart = True
    brm._simulation_arguments = {"active_realizations": active_realizations}
    brm.has_failed_realizations = lambda: False
    return brm


class MockTracker:
    def __init__(self, events) -> None:
        self._events = events

    def track(self):
        for event in self._events:
            yield event
            time.sleep(0.1)

    def reset(self):
        pass

    def request_termination(self):
        pass


@pytest.fixture
def mock_tracker():
    def _make_mock_tracker(events):
        return MockTracker(events)

    return _make_mock_tracker
