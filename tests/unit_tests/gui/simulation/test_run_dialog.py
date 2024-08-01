import os
from queue import SimpleQueue
from unittest.mock import MagicMock, Mock, patch

import pytest
from pytestqt.qtbot import QtBot
from qtpy import QtWidgets
from qtpy.QtCore import Qt, QTimer
from qtpy.QtWidgets import QComboBox, QToolButton, QWidget

import ert
from ert.config import ErtConfig
from ert.ensemble_evaluator import state
from ert.ensemble_evaluator.event import (
    EndEvent,
    FullSnapshotEvent,
    SnapshotUpdateEvent,
)
from ert.ensemble_evaluator.snapshot import PartialSnapshot, SnapshotBuilder
from ert.gui.ertnotifier import ErtNotifier
from ert.gui.main import GUILogHandler, _setup_main_window
from ert.gui.simulation.ensemble_experiment_panel import EnsembleExperimentPanel
from ert.gui.simulation.experiment_panel import ExperimentPanel
from ert.gui.simulation.run_dialog import RunDialog
from ert.gui.simulation.view.realization import RealizationWidget
from ert.gui.tools.file import FileDialog
from ert.run_models import BaseRunModel
from ert.run_models.ensemble_experiment import EnsembleExperiment
from ert.services import StorageService
from ert.storage import open_storage
from tests.unit_tests.gui.simulation.test_run_path_dialog import handle_run_path_dialog

from ..conftest import wait_for_child


@pytest.fixture
def run_model():
    run_model = MagicMock(spec=BaseRunModel)
    run_model.hasRunFailed.return_value = False
    run_model.getFailMessage.return_value = ""
    run_model.get_runtime.return_value = 1
    run_model.support_restart = True
    return run_model


@pytest.fixture
def event_queue():
    return SimpleQueue()


@pytest.fixture
def notifier():
    return MagicMock(spec=ErtNotifier)


@pytest.fixture
def run_dialog(qtbot: QtBot, run_model, event_queue, notifier):
    run_dialog = RunDialog("mock.ert", run_model, event_queue, notifier)
    qtbot.addWidget(run_dialog)
    return run_dialog


def test_that_done_button_is_not_hidden_when_the_end_event_is_given(
    qtbot: QtBot, run_dialog, event_queue
):
    run_dialog.run_experiment()
    event_queue.put(EndEvent(failed=False, failed_msg=""))
    qtbot.waitUntil(lambda: not run_dialog.done_button.isHidden(), timeout=1000)
    assert not run_dialog.done_button.isHidden()
    qtbot.mouseClick(run_dialog.done_button, Qt.LeftButton)


def test_terminating_experiment_shows_a_confirmation_dialog(
    qtbot: QtBot, run_dialog, event_queue
):
    run_dialog.run_experiment()
    event_queue.put(EndEvent(failed=False, failed_msg=""))

    with qtbot.waitSignal(run_dialog.finished, timeout=30000):

        def handle_dialog():
            qtbot.waitUntil(
                lambda: run_dialog.findChild(QtWidgets.QMessageBox) is not None
            )
            confirm_terminate_dialog = run_dialog.findChild(QtWidgets.QMessageBox)
            assert isinstance(confirm_terminate_dialog, QtWidgets.QMessageBox)
            dialog_buttons = confirm_terminate_dialog.findChild(
                QtWidgets.QDialogButtonBox
            ).buttons()
            yes_button = [b for b in dialog_buttons if "Yes" in b.text()][0]
            qtbot.mouseClick(yes_button, Qt.LeftButton)

        QTimer.singleShot(100, handle_dialog)
        qtbot.mouseClick(run_dialog.kill_button, Qt.LeftButton)


def test_detail_view_toggle(qtbot: QtBot, run_dialog: RunDialog):
    details_toggled = "Hide" in run_dialog.show_details_button.text()
    qtbot.mouseClick(run_dialog.show_details_button, Qt.LeftButton)
    keyword = "Show" if details_toggled else "Hide"
    qtbot.waitUntil(lambda: keyword in run_dialog.show_details_button.text())


def test_run_dialog_polls_run_model_for_runtime(
    qtbot: QtBot, run_dialog: RunDialog, run_model, notifier, event_queue
):
    run_dialog.run_experiment()
    notifier.set_is_simulation_running.assert_called_with(True)
    qtbot.waitUntil(
        lambda: run_model.get_runtime.called, timeout=run_dialog._RUN_TIME_POLL_RATE * 2
    )
    event_queue.put(EndEvent(failed=False, failed_msg=""))
    qtbot.waitUntil(lambda: not run_dialog.done_button.isHidden())
    run_dialog.close()


def test_large_snapshot(
    large_snapshot,
    qtbot: QtBot,
    run_dialog: RunDialog,
    event_queue,
    timeout_per_iter=5000,
):
    events = [
        FullSnapshotEvent(
            snapshot=large_snapshot,
            phase_name="Foo",
            current_phase=0,
            total_phases=1,
            progress=0.5,
            realization_count=4,
            status_count={"Finished": 2, "Unknown": 2},
            iteration=0,
        ),
        FullSnapshotEvent(
            snapshot=large_snapshot,
            phase_name="Foo",
            current_phase=0,
            total_phases=1,
            progress=0.5,
            realization_count=4,
            status_count={"Finished": 2, "Unknown": 2},
            iteration=1,
        ),
        EndEvent(failed=False, failed_msg=""),
    ]

    run_dialog.run_experiment()
    for event in events:
        event_queue.put(event)

    qtbot.waitUntil(
        lambda: run_dialog._total_progress_bar.value() == 100,
        timeout=timeout_per_iter * 3,
    )
    qtbot.mouseClick(run_dialog.show_details_button, Qt.LeftButton)
    qtbot.waitUntil(
        lambda: run_dialog._tab_widget.count() == 2, timeout=timeout_per_iter
    )
    qtbot.waitUntil(
        lambda: not run_dialog.done_button.isHidden(), timeout=timeout_per_iter
    )


@pytest.mark.parametrize(
    "events,tab_widget_count",
    [
        pytest.param(
            [
                FullSnapshotEvent(
                    snapshot=(
                        SnapshotBuilder()
                        .add_forward_model(
                            forward_model_id="0",
                            index="0",
                            name="job_0",
                            status=state.FORWARD_MODEL_STATE_START,
                        )
                        .build(["0"], state.REALIZATION_STATE_UNKNOWN)
                    ),
                    phase_name="Foo",
                    current_phase=0,
                    total_phases=1,
                    progress=0.25,
                    realization_count=4,
                    status_count={"Finished": 1, "Pending": 1, "Unknown": 2},
                    iteration=0,
                ),
                SnapshotUpdateEvent(
                    partial_snapshot=PartialSnapshot(
                        SnapshotBuilder().build(
                            [], status=state.REALIZATION_STATE_FINISHED
                        )
                    ),
                    phase_name="Foo",
                    current_phase=0,
                    total_phases=1,
                    progress=0.5,
                    realization_count=4,
                    status_count={"Finished": 2, "Unknown": 2},
                    iteration=0,
                ),
                EndEvent(failed=False, failed_msg=""),
            ],
            1,
            id="real_less_partial",
        ),
        pytest.param(
            [
                FullSnapshotEvent(
                    snapshot=(
                        SnapshotBuilder()
                        .add_forward_model(
                            forward_model_id="0",
                            index="0",
                            name="job_0",
                            max_memory_usage="1000",
                            current_memory_usage="500",
                            status=state.FORWARD_MODEL_STATE_START,
                        )
                        .build(["0"], state.REALIZATION_STATE_UNKNOWN)
                    ),
                    phase_name="Foo",
                    current_phase=0,
                    total_phases=1,
                    progress=0.25,
                    realization_count=4,
                    status_count={"Finished": 1, "Pending": 1, "Unknown": 2},
                    iteration=0,
                ),
                SnapshotUpdateEvent(
                    partial_snapshot=PartialSnapshot(
                        SnapshotBuilder().build(
                            ["0"], status=state.REALIZATION_STATE_FINISHED
                        )
                    ),
                    phase_name="Foo",
                    current_phase=0,
                    total_phases=1,
                    progress=0.5,
                    realization_count=4,
                    status_count={"Finished": 2, "Unknown": 2},
                    iteration=0,
                ),
                EndEvent(failed=False, failed_msg=""),
            ],
            1,
            id="jobless_partial",
        ),
        pytest.param(
            [
                FullSnapshotEvent(
                    snapshot=(
                        SnapshotBuilder()
                        .add_forward_model(
                            forward_model_id="0",
                            index="0",
                            name="job_0",
                            status=state.FORWARD_MODEL_STATE_START,
                        )
                        .add_forward_model(
                            forward_model_id="1",
                            index="1",
                            name="job_1",
                            status=state.FORWARD_MODEL_STATE_START,
                        )
                        .build(["0", "1"], state.REALIZATION_STATE_UNKNOWN)
                    ),
                    phase_name="Foo",
                    current_phase=0,
                    total_phases=1,
                    progress=0.25,
                    realization_count=4,
                    status_count={"Finished": 1, "Pending": 2, "Unknown": 1},
                    iteration=0,
                ),
                SnapshotUpdateEvent(
                    partial_snapshot=PartialSnapshot(
                        SnapshotBuilder()
                        .add_forward_model(
                            forward_model_id="0",
                            index="0",
                            status=state.FORWARD_MODEL_STATE_FINISHED,
                            name="job_0",
                        )
                        .build(["1"], status=state.REALIZATION_STATE_RUNNING)
                    ),
                    phase_name="Foo",
                    current_phase=0,
                    total_phases=1,
                    progress=0.5,
                    realization_count=4,
                    status_count={"Finished": 2, "Running": 1, "Unknown": 1},
                    iteration=0,
                ),
                SnapshotUpdateEvent(
                    partial_snapshot=PartialSnapshot(
                        SnapshotBuilder()
                        .add_forward_model(
                            forward_model_id="1",
                            index="1",
                            status=state.FORWARD_MODEL_STATE_FAILURE,
                            name="job_1",
                        )
                        .build(["0"], status=state.REALIZATION_STATE_FAILED)
                    ),
                    phase_name="Foo",
                    current_phase=0,
                    total_phases=1,
                    progress=0.5,
                    realization_count=4,
                    status_count={"Finished": 2, "Failed": 1, "Unknown": 1},
                    iteration=0,
                ),
                EndEvent(failed=False, failed_msg=""),
            ],
            1,
            id="two_job_updates_over_two_partials",
        ),
        pytest.param(
            [
                FullSnapshotEvent(
                    snapshot=(
                        SnapshotBuilder()
                        .add_forward_model(
                            forward_model_id="0",
                            index="0",
                            name="job_0",
                            status=state.FORWARD_MODEL_STATE_START,
                        )
                        .build(["0"], state.REALIZATION_STATE_UNKNOWN)
                    ),
                    phase_name="Foo",
                    current_phase=0,
                    total_phases=1,
                    progress=0.25,
                    realization_count=4,
                    status_count={"Pending": 1, "Unknown": 3},
                    iteration=0,
                ),
                FullSnapshotEvent(
                    snapshot=(
                        SnapshotBuilder()
                        .add_forward_model(
                            forward_model_id="0",
                            index="0",
                            name="job_0",
                            status=state.FORWARD_MODEL_STATE_START,
                        )
                        .build(["0"], state.REALIZATION_STATE_UNKNOWN)
                    ),
                    phase_name="Foo",
                    current_phase=0,
                    total_phases=1,
                    progress=0.5,
                    realization_count=4,
                    status_count={"Finished": 1, "Pending": 1, "Unknown": 2},
                    iteration=1,
                ),
                EndEvent(failed=False, failed_msg=""),
            ],
            2,
            id="two_iterations",
        ),
    ],
)
def test_run_dialog(events, tab_widget_count, qtbot: QtBot, run_dialog, event_queue):
    run_dialog.run_experiment()
    for event in events:
        event_queue.put(event)

    qtbot.mouseClick(run_dialog.show_details_button, Qt.LeftButton)
    qtbot.waitUntil(
        lambda: run_dialog._tab_widget.count() == tab_widget_count, timeout=5000
    )
    qtbot.waitUntil(lambda: not run_dialog.done_button.isHidden(), timeout=5000)


def test_that_run_dialog_can_be_closed_while_file_plot_is_open(
    snake_oil_case_storage: ErtConfig, qtbot: QtBot
):
    """
    This is a regression test for a crash happening when
    closing the RunDialog with a file open.
    """

    snake_oil_case = snake_oil_case_storage
    args_mock = Mock()
    args_mock.config = "snake_oil.ert"

    with StorageService.init_service(
        project=os.path.abspath(snake_oil_case.ens_path),
    ), open_storage(snake_oil_case.ens_path, mode="w") as storage:
        gui = _setup_main_window(snake_oil_case, args_mock, GUILogHandler(), storage)
        experiment_panel = gui.findChild(ExperimentPanel)

        run_experiment = experiment_panel.findChild(QWidget, name="run_experiment")
        assert run_experiment
        assert isinstance(run_experiment, QToolButton)

        QTimer.singleShot(
            1000, lambda: handle_run_path_dialog(gui, qtbot, delete_run_path=True)
        )
        qtbot.mouseClick(run_experiment, Qt.LeftButton)

        qtbot.waitUntil(lambda: gui.findChild(RunDialog) is not None, timeout=5000)
        run_dialog = gui.findChild(RunDialog)
        qtbot.mouseClick(run_dialog.show_details_button, Qt.LeftButton)
        qtbot.waitUntil(run_dialog.done_button.isVisible, timeout=100000)
        job_overview = run_dialog._job_overview

        qtbot.waitUntil(job_overview.isVisible, timeout=20000)
        qtbot.waitUntil(run_dialog.done_button.isVisible, timeout=200000)

        realization_widget = run_dialog.findChild(RealizationWidget)

        click_pos = realization_widget._real_view.rectForIndex(
            realization_widget._real_list_model.index(0, 0)
        ).center()

        with qtbot.waitSignal(realization_widget.currentChanged, timeout=30000):
            qtbot.mouseClick(
                realization_widget._real_view.viewport(),
                Qt.LeftButton,
                pos=click_pos,
            )

        click_pos = job_overview.visualRect(job_overview.model().index(0, 4)).center()
        qtbot.mouseClick(job_overview.viewport(), Qt.LeftButton, pos=click_pos)

        qtbot.waitUntil(run_dialog.findChild(FileDialog).isVisible, timeout=30000)

        with qtbot.waitSignal(run_dialog.accepted, timeout=30000):
            run_dialog.close()  # Close the run dialog by pressing 'x' close button

        # Ensure that once the run dialog is closed
        # another simulation can be started
        assert run_experiment.isEnabled()


@pytest.mark.parametrize(
    "events,tab_widget_count",
    [
        pytest.param(
            [
                FullSnapshotEvent(
                    snapshot=(
                        SnapshotBuilder()
                        .add_forward_model(
                            forward_model_id="0",
                            index="0",
                            name="job_0",
                            status=state.FORWARD_MODEL_STATE_START,
                        )
                        .build(["0"], state.REALIZATION_STATE_UNKNOWN)
                    ),
                    phase_name="Foo",
                    current_phase=0,
                    total_phases=1,
                    progress=0.25,
                    realization_count=4,
                    status_count={"Finished": 1, "Pending": 1, "Unknown": 2},
                    iteration=0,
                ),
                SnapshotUpdateEvent(
                    partial_snapshot=PartialSnapshot(
                        SnapshotBuilder()
                        .add_forward_model(
                            forward_model_id="0",
                            index="0",
                            status=state.FORWARD_MODEL_STATE_RUNNING,
                            current_memory_usage="45000",
                            max_memory_usage="55000",
                            name="job_0",
                        )
                        .build(["0"], status=state.REALIZATION_STATE_RUNNING)
                    ),
                    phase_name="Foo",
                    current_phase=0,
                    total_phases=1,
                    progress=0.5,
                    realization_count=4,
                    status_count={"Finished": 2, "Running": 1, "Unknown": 1},
                    iteration=0,
                ),
                SnapshotUpdateEvent(
                    partial_snapshot=PartialSnapshot(
                        SnapshotBuilder()
                        .add_forward_model(
                            forward_model_id="0",
                            index="0",
                            status=state.FORWARD_MODEL_STATE_FINISHED,
                            name="job_0",
                            current_memory_usage="50000",
                            max_memory_usage="60000",
                        )
                        .build(["0"], status=state.REALIZATION_STATE_FINISHED)
                    ),
                    phase_name="Foo",
                    current_phase=0,
                    total_phases=1,
                    progress=1,
                    realization_count=4,
                    status_count={"Finished": 4},
                    iteration=0,
                ),
                EndEvent(failed=False, failed_msg=""),
            ],
            1,
            id="running_job_with_memory_usage",
        ),
    ],
)
def test_run_dialog_memory_usage_showing(
    events, tab_widget_count, qtbot: QtBot, event_queue, run_dialog
):
    run_dialog.run_experiment()
    for event in events:
        event_queue.put(event)

    qtbot.mouseClick(run_dialog.show_details_button, Qt.LeftButton)
    qtbot.waitUntil(
        lambda: run_dialog._tab_widget.count() == tab_widget_count, timeout=5000
    )
    qtbot.waitUntil(lambda: not run_dialog.done_button.isHidden(), timeout=5000)

    # This is the container of realization boxes
    realization_box = run_dialog._tab_widget.widget(0)
    assert type(realization_box) == RealizationWidget
    # Click the first realization box
    qtbot.mouseClick(realization_box, Qt.LeftButton)
    job_model = run_dialog._job_overview.model()
    assert job_model._real == 0

    job_number = 0
    current_memory_column_index = 6
    max_memory_column_index = 7

    current_memory_column_proxy_index = job_model.index(
        job_number, current_memory_column_index
    )
    current_memory_value = job_model.data(
        current_memory_column_proxy_index, Qt.DisplayRole
    )
    assert current_memory_value == "50.00 kB"

    max_memory_column_proxy_index = job_model.index(job_number, max_memory_column_index)
    max_memory_value = job_model.data(max_memory_column_proxy_index, Qt.DisplayRole)
    assert max_memory_value == "60.00 kB"


@pytest.mark.usefixtures("use_tmpdir", "set_site_config")
def test_that_gui_runs_a_minimal_example(qtbot: QtBot, storage):
    """
    This is a regression test for a crash happening when clicking show details
    when running a minimal example.
    """
    config_file = "minimal_config.ert"
    with open(config_file, "w", encoding="utf-8") as f:
        f.write("NUM_REALIZATIONS 1")
    args_mock = Mock()
    args_mock.config = config_file

    ert_config = ErtConfig.from_file(config_file)
    with StorageService.init_service(
        project=os.path.abspath(ert_config.ens_path),
    ):
        gui = _setup_main_window(ert_config, args_mock, GUILogHandler(), storage)
        qtbot.addWidget(gui)
        run_experiment = gui.findChild(QToolButton, name="run_experiment")

        qtbot.mouseClick(run_experiment, Qt.LeftButton)

        qtbot.waitUntil(lambda: gui.findChild(RunDialog) is not None)
        run_dialog = gui.findChild(RunDialog)
        qtbot.mouseClick(run_dialog.show_details_button, Qt.LeftButton)
        qtbot.waitUntil(run_dialog.done_button.isVisible, timeout=200000)


@pytest.mark.usefixtures("use_tmpdir")
def test_that_exception_in_base_run_model_is_handled(qtbot: QtBot, storage):
    config_file = "minimal_config.ert"
    with open(config_file, "w", encoding="utf-8") as f:
        f.write("NUM_REALIZATIONS 1")
    args_mock = Mock()
    args_mock.config = config_file

    ert_config = ErtConfig.from_file(config_file)
    with StorageService.init_service(
        project=os.path.abspath(ert_config.ens_path),
    ), patch.object(
        ert.run_models.SingleTestRun,
        "run_experiment",
        MagicMock(side_effect=ValueError("I failed :(")),
    ):
        gui = _setup_main_window(ert_config, args_mock, GUILogHandler(), storage)
        qtbot.addWidget(gui)
        run_experiment = gui.findChild(QToolButton, name="run_experiment")

        def handle_error_dialog(run_dialog):
            error_dialog = run_dialog.fail_msg_box
            assert error_dialog
            text = error_dialog.details_text.toPlainText()
            assert "I failed :(" in text
            qtbot.mouseClick(error_dialog.box.buttons()[0], Qt.LeftButton)

        qtbot.mouseClick(run_experiment, Qt.LeftButton)

        run_dialog = wait_for_child(gui, qtbot, RunDialog)
        qtbot.mouseClick(run_dialog.show_details_button, Qt.LeftButton)

        QTimer.singleShot(100, lambda: handle_error_dialog(run_dialog))
        qtbot.waitUntil(run_dialog.done_button.isVisible, timeout=200000)


def test_that_stdout_and_stderr_buttons_react_to_file_content(
    snake_oil_case_storage: ErtConfig, qtbot: QtBot
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

        run_experiment = experiment_panel.findChild(QWidget, name="run_experiment")
        assert run_experiment
        assert isinstance(run_experiment, QToolButton)

        QTimer.singleShot(
            1000, lambda: handle_run_path_dialog(gui, qtbot, delete_run_path=True)
        )
        qtbot.mouseClick(run_experiment, Qt.LeftButton)

        qtbot.waitUntil(lambda: gui.findChild(RunDialog) is not None, timeout=5000)
        run_dialog = gui.findChild(RunDialog)
        qtbot.mouseClick(run_dialog.show_details_button, Qt.LeftButton)
        qtbot.waitUntil(run_dialog.done_button.isVisible, timeout=100000)
        job_overview = run_dialog._job_overview

        qtbot.waitUntil(job_overview.isVisible, timeout=20000)

        realization_widget = run_dialog.findChild(RealizationWidget)

        click_pos = realization_widget._real_view.rectForIndex(
            realization_widget._real_list_model.index(0, 0)
        ).center()

        with qtbot.waitSignal(realization_widget.currentChanged, timeout=30000):
            qtbot.mouseClick(
                realization_widget._real_view.viewport(),
                Qt.LeftButton,
                pos=click_pos,
            )

        job_stdout = job_overview.model().index(0, 4)
        job_stderr = job_overview.model().index(0, 5)

        assert job_stdout.data(Qt.ItemDataRole.DisplayRole) == "View"
        assert job_stderr.data(Qt.ItemDataRole.DisplayRole) == "-"
        assert job_stdout.data(Qt.ItemDataRole.ForegroundRole) == Qt.GlobalColor.blue
        assert job_stderr.data(Qt.ItemDataRole.ForegroundRole) == None

        assert job_stdout.data(Qt.ItemDataRole.FontRole).underline() == True
        assert job_stderr.data(Qt.ItemDataRole.FontRole) == None

        click_pos = job_overview.visualRect(job_stdout).center()

        qtbot.mouseClick(job_overview.viewport(), Qt.LeftButton, pos=click_pos)

        qtbot.waitUntil(run_dialog.findChild(FileDialog).isVisible, timeout=30000)

        with qtbot.waitSignal(run_dialog.accepted, timeout=30000):
            run_dialog.close()
