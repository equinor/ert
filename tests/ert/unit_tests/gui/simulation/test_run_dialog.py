import os
from queue import SimpleQueue
from unittest.mock import MagicMock, Mock, patch

import pytest
from pytestqt.qtbot import QtBot
from qtpy import QtWidgets
from qtpy.QtCore import Qt, QTimer
from qtpy.QtWidgets import (
    QApplication,
    QComboBox,
    QLabel,
    QPushButton,
    QToolButton,
    QWidget,
)

import ert
from ert.config import ErtConfig
from ert.ensemble_evaluator import state
from ert.ensemble_evaluator.event import (
    EndEvent,
    FullSnapshotEvent,
    SnapshotUpdateEvent,
)
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
from tests.ert import SnapshotBuilder
from tests.ert.ui_tests.gui.conftest import wait_for_child
from tests.ert.unit_tests.gui.simulation.test_run_path_dialog import (
    handle_run_path_dialog,
)


@pytest.fixture
def run_model():
    run_model = MagicMock(spec=BaseRunModel)
    run_model.format_error.return_value = ""
    run_model.get_runtime.return_value = 1
    run_model.support_restart = True
    return run_model


@pytest.fixture
def event_queue():
    return SimpleQueue()


@pytest.fixture
def notifier():
    notifier = MagicMock(spec=ErtNotifier)
    notifier.is_simulation_running = False
    return notifier


@pytest.fixture
def run_dialog(qtbot: QtBot, run_model, event_queue, notifier):
    run_dialog = RunDialog("mock.ert", run_model, event_queue, notifier)
    qtbot.addWidget(run_dialog)
    return run_dialog


def test_that_done_button_is_not_hidden_when_the_end_event_is_given(
    qtbot: QtBot, run_dialog, event_queue
):
    run_dialog.run_experiment()
    event_queue.put(EndEvent(failed=False, msg=""))
    qtbot.waitUntil(lambda: not run_dialog.done_button.isHidden(), timeout=1000)
    assert not run_dialog.done_button.isHidden()
    qtbot.mouseClick(run_dialog.done_button, Qt.LeftButton)


def test_terminating_experiment_shows_a_confirmation_dialog(
    qtbot: QtBot, run_dialog, event_queue
):
    run_dialog.run_experiment()
    event_queue.put(EndEvent(failed=False, msg=""))

    with qtbot.waitSignal(run_dialog.finished, timeout=30000):

        def handle_dialog():
            confirm_terminate_dialog = wait_for_child(
                run_dialog, qtbot, QtWidgets.QMessageBox
            )
            dialog_buttons = confirm_terminate_dialog.findChild(
                QtWidgets.QDialogButtonBox
            ).buttons()
            yes_button = next(b for b in dialog_buttons if "Yes" in b.text())
            qtbot.mouseClick(yes_button, Qt.LeftButton)

        QTimer.singleShot(100, handle_dialog)
        qtbot.mouseClick(run_dialog.kill_button, Qt.LeftButton)


@pytest.mark.integration_test
def test_run_dialog_polls_run_model_for_runtime(
    qtbot: QtBot, run_dialog: RunDialog, run_model, notifier, event_queue
):
    run_dialog.run_experiment()
    notifier.set_is_simulation_running.assert_called_with(True)
    qtbot.waitUntil(
        lambda: run_model.get_runtime.called, timeout=run_dialog._RUN_TIME_POLL_RATE * 2
    )
    event_queue.put(EndEvent(failed=False, msg=""))
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
            iteration_label="Foo",
            current_iteration=0,
            total_iterations=1,
            progress=0.5,
            realization_count=4,
            status_count={"Finished": 2, "Unknown": 2},
            iteration=0,
        ),
        FullSnapshotEvent(
            snapshot=large_snapshot,
            iteration_label="Foo",
            current_iteration=0,
            total_iterations=1,
            progress=0.5,
            realization_count=4,
            status_count={"Finished": 2, "Unknown": 2},
            iteration=1,
        ),
        EndEvent(failed=False, msg=""),
    ]

    run_dialog.run_experiment()
    for event in events:
        event_queue.put(event)

    qtbot.waitUntil(
        lambda: run_dialog._total_progress_bar.value() == 100,
        timeout=timeout_per_iter * 3,
    )
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
                        .add_fm_step(
                            fm_step_id="0",
                            index="0",
                            name="fm_step_0",
                            status=state.FORWARD_MODEL_STATE_START,
                        )
                        .build(["0"], state.REALIZATION_STATE_UNKNOWN)
                    ),
                    iteration_label="Foo",
                    current_iteration=0,
                    total_iterations=1,
                    progress=0.25,
                    realization_count=4,
                    status_count={"Finished": 1, "Pending": 1, "Unknown": 2},
                    iteration=0,
                ),
                SnapshotUpdateEvent(
                    snapshot=SnapshotBuilder().build(
                        [], status=state.REALIZATION_STATE_FINISHED
                    ),
                    iteration_label="Foo",
                    current_iteration=0,
                    total_iterations=1,
                    progress=0.5,
                    realization_count=4,
                    status_count={"Finished": 2, "Unknown": 2},
                    iteration=0,
                ),
                EndEvent(failed=False, msg=""),
            ],
            1,
            id="real_less_partial",
        ),
        pytest.param(
            [
                FullSnapshotEvent(
                    snapshot=(
                        SnapshotBuilder()
                        .add_fm_step(
                            fm_step_id="0",
                            index="0",
                            name="fm_step_0",
                            max_memory_usage="1000",
                            current_memory_usage="500",
                            status=state.FORWARD_MODEL_STATE_START,
                        )
                        .build(["0"], state.REALIZATION_STATE_UNKNOWN)
                    ),
                    iteration_label="Foo",
                    current_iteration=0,
                    total_iterations=1,
                    progress=0.25,
                    realization_count=4,
                    status_count={"Finished": 1, "Pending": 1, "Unknown": 2},
                    iteration=0,
                ),
                SnapshotUpdateEvent(
                    snapshot=SnapshotBuilder().build(
                        ["0"], status=state.REALIZATION_STATE_FINISHED
                    ),
                    iteration_label="Foo",
                    current_iteration=0,
                    total_iterations=1,
                    progress=0.5,
                    realization_count=4,
                    status_count={"Finished": 2, "Unknown": 2},
                    iteration=0,
                ),
                EndEvent(failed=False, msg=""),
            ],
            1,
            id="fm_stepless_partial",
        ),
        pytest.param(
            [
                FullSnapshotEvent(
                    snapshot=(
                        SnapshotBuilder()
                        .add_fm_step(
                            fm_step_id="0",
                            index="0",
                            name="fm_step_0",
                            status=state.FORWARD_MODEL_STATE_START,
                        )
                        .add_fm_step(
                            fm_step_id="1",
                            index="1",
                            name="fm_step_1",
                            status=state.FORWARD_MODEL_STATE_START,
                        )
                        .build(["0", "1"], state.REALIZATION_STATE_UNKNOWN)
                    ),
                    iteration_label="Foo",
                    current_iteration=0,
                    total_iterations=1,
                    progress=0.25,
                    realization_count=4,
                    status_count={"Finished": 1, "Pending": 2, "Unknown": 1},
                    iteration=0,
                ),
                SnapshotUpdateEvent(
                    snapshot=SnapshotBuilder()
                    .add_fm_step(
                        fm_step_id="0",
                        index="0",
                        status=state.FORWARD_MODEL_STATE_FINISHED,
                        name="fm_step_0",
                    )
                    .build(["1"], status=state.REALIZATION_STATE_RUNNING),
                    iteration_label="Foo",
                    current_iteration=0,
                    total_iterations=1,
                    progress=0.5,
                    realization_count=4,
                    status_count={"Finished": 2, "Running": 1, "Unknown": 1},
                    iteration=0,
                ),
                SnapshotUpdateEvent(
                    snapshot=SnapshotBuilder()
                    .add_fm_step(
                        fm_step_id="1",
                        index="1",
                        status=state.FORWARD_MODEL_STATE_FAILURE,
                        name="fm_step_1",
                    )
                    .build(["0"], status=state.REALIZATION_STATE_FAILED),
                    iteration_label="Foo",
                    current_iteration=0,
                    total_iterations=1,
                    progress=0.5,
                    realization_count=4,
                    status_count={"Finished": 2, "Failed": 1, "Unknown": 1},
                    iteration=0,
                ),
                EndEvent(failed=False, msg=""),
            ],
            1,
            id="two_fm_step_updates_over_two_partials",
        ),
        pytest.param(
            [
                FullSnapshotEvent(
                    snapshot=(
                        SnapshotBuilder()
                        .add_fm_step(
                            fm_step_id="0",
                            index="0",
                            name="fm_step_0",
                            status=state.FORWARD_MODEL_STATE_START,
                        )
                        .build(["0"], state.REALIZATION_STATE_UNKNOWN)
                    ),
                    iteration_label="Foo",
                    current_iteration=0,
                    total_iterations=1,
                    progress=0.25,
                    realization_count=4,
                    status_count={"Pending": 1, "Unknown": 3},
                    iteration=0,
                ),
                FullSnapshotEvent(
                    snapshot=(
                        SnapshotBuilder()
                        .add_fm_step(
                            fm_step_id="0",
                            index="0",
                            name="fm_step_0",
                            status=state.FORWARD_MODEL_STATE_START,
                        )
                        .build(["0"], state.REALIZATION_STATE_UNKNOWN)
                    ),
                    iteration_label="Foo",
                    current_iteration=0,
                    total_iterations=1,
                    progress=0.5,
                    realization_count=4,
                    status_count={"Finished": 1, "Pending": 1, "Unknown": 2},
                    iteration=1,
                ),
                EndEvent(failed=False, msg=""),
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

    qtbot.waitUntil(
        lambda: run_dialog._tab_widget.count() == tab_widget_count, timeout=5000
    )
    qtbot.waitUntil(lambda: not run_dialog.done_button.isHidden(), timeout=5000)


@pytest.mark.parametrize(
    "events,tab_widget_count",
    [
        pytest.param(
            [
                FullSnapshotEvent(
                    snapshot=(
                        SnapshotBuilder()
                        .add_fm_step(
                            fm_step_id="0",
                            index="0",
                            name="fm_step_0",
                            status=state.FORWARD_MODEL_STATE_START,
                        )
                        .build(["0"], state.REALIZATION_STATE_UNKNOWN)
                    ),
                    iteration_label="Foo",
                    current_iteration=0,
                    total_iterations=1,
                    progress=0.25,
                    realization_count=4,
                    status_count={"Finished": 1, "Pending": 1, "Unknown": 2},
                    iteration=0,
                ),
                SnapshotUpdateEvent(
                    snapshot=SnapshotBuilder()
                    .add_fm_step(
                        fm_step_id="0",
                        index="0",
                        status=state.FORWARD_MODEL_STATE_RUNNING,
                        current_memory_usage="45000",
                        max_memory_usage="55000",
                        name="fm_step_0",
                    )
                    .build(["0"], status=state.REALIZATION_STATE_RUNNING),
                    iteration_label="Foo",
                    current_iteration=0,
                    total_iterations=1,
                    progress=0.5,
                    realization_count=4,
                    status_count={"Finished": 2, "Running": 1, "Unknown": 1},
                    iteration=0,
                ),
                SnapshotUpdateEvent(
                    snapshot=SnapshotBuilder()
                    .add_fm_step(
                        fm_step_id="0",
                        index="0",
                        status=state.FORWARD_MODEL_STATE_FINISHED,
                        name="fm_step_0",
                        current_memory_usage="50000",
                        max_memory_usage="60000",
                    )
                    .build(["0"], status=state.REALIZATION_STATE_FINISHED),
                    iteration_label="Foo",
                    current_iteration=0,
                    total_iterations=1,
                    progress=1,
                    realization_count=4,
                    status_count={"Finished": 4},
                    iteration=0,
                ),
                EndEvent(failed=False, msg=""),
            ],
            1,
            id="running_fm_step_with_memory_usage",
        ),
    ],
)
def test_run_dialog_memory_usage_showing(
    events, tab_widget_count, qtbot: QtBot, event_queue, run_dialog
):
    run_dialog.run_experiment()
    for event in events:
        event_queue.put(event)

    qtbot.waitUntil(
        lambda: run_dialog._tab_widget.count() == tab_widget_count, timeout=5000
    )
    qtbot.waitUntil(lambda: not run_dialog.done_button.isHidden(), timeout=5000)

    # This is the container of realization boxes
    realization_box = run_dialog._tab_widget.widget(0)
    assert type(realization_box) == RealizationWidget
    # Click the first realization box
    qtbot.mouseClick(realization_box, Qt.LeftButton)
    fm_step_model = run_dialog._fm_step_overview.model()
    assert fm_step_model._real == 0

    fm_step_number = 0
    max_memory_column_index = 6

    max_memory_column_proxy_index = fm_step_model.index(
        fm_step_number, max_memory_column_index
    )
    max_memory_value = fm_step_model.data(max_memory_column_proxy_index, Qt.DisplayRole)
    assert max_memory_value == "60.00 KB"


@pytest.mark.parametrize(
    "events, tab_widget_count, expected_host_info",
    [
        pytest.param(
            [
                FullSnapshotEvent(
                    snapshot=(
                        SnapshotBuilder()
                        .add_fm_step(
                            fm_step_id="0",
                            index="0",
                            name="fm_step_0",
                            status=state.FORWARD_MODEL_STATE_START,
                        )
                        .build(
                            ["0"],
                            status=state.REALIZATION_STATE_UNKNOWN,
                            exec_hosts="COMP_01",
                        )
                    ),
                    iteration_label="Foo",
                    current_iteration=0,
                    total_iterations=1,
                    progress=0.25,
                    realization_count=4,
                    status_count={"Finished": 1, "Pending": 1, "Unknown": 2},
                    iteration=0,
                ),
                EndEvent(failed=False, msg=""),
            ],
            1,
            ", assigned to host: COMP_01",
            id="Simulation where exec_host present",
        ),
        pytest.param(
            [
                FullSnapshotEvent(
                    snapshot=(
                        SnapshotBuilder()
                        .add_fm_step(
                            fm_step_id="0",
                            index="0",
                            name="fm_step_0",
                            status=state.FORWARD_MODEL_STATE_START,
                        )
                        .build(["0"], status=state.REALIZATION_STATE_UNKNOWN)
                    ),
                    iteration_label="Foo",
                    current_iteration=0,
                    total_iterations=1,
                    progress=0.25,
                    realization_count=4,
                    status_count={"Finished": 1, "Pending": 1, "Unknown": 2},
                    iteration=0,
                ),
                EndEvent(failed=False, msg=""),
            ],
            1,
            "",
            id="Simulation where exec_host not present",
        ),
    ],
)
def test_run_dialog_fm_label_show_correct_info(
    events, tab_widget_count, expected_host_info, qtbot: QtBot, event_queue, run_dialog
):
    run_dialog.run_experiment()
    for event in events:
        event_queue.put(event)

    qtbot.waitUntil(
        lambda: run_dialog._tab_widget.count() == tab_widget_count, timeout=5000
    )
    qtbot.waitUntil(lambda: not run_dialog.done_button.isHidden(), timeout=5000)

    # This is the container of realization boxes
    realization_box = run_dialog._tab_widget.widget(0)
    assert type(realization_box) == RealizationWidget
    # Click the first realization box
    qtbot.mouseClick(realization_box, Qt.LeftButton)
    fm_step_model = run_dialog._fm_step_overview.model()
    assert fm_step_model._real == 0

    fm_step_label = run_dialog.findChild(QLabel, name="fm_step_label")
    assert not fm_step_label.text()

    realization_box._item_clicked(run_dialog._fm_step_overview.model().index(0, 0))
    assert (
        fm_step_label.text() == f"Realization id 0 in iteration 0{expected_host_info}"
    )


@pytest.mark.integration_test
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
            qtbot.waitUntil(
                lambda: run_dialog.fail_msg_box is not None,
                timeout=20000,
            )
            error_dialog = run_dialog.fail_msg_box
            assert error_dialog
            text = error_dialog.details_text.toPlainText()
            assert "I failed :(" in text
            qtbot.mouseClick(error_dialog.box.buttons()[0], Qt.LeftButton)

        simulation_mode_combo = gui.findChild(QComboBox)
        simulation_mode_combo.setCurrentText("Single realization test-run")
        qtbot.mouseClick(run_experiment, Qt.LeftButton)
        run_dialog = wait_for_child(gui, qtbot, RunDialog)

        QTimer.singleShot(100, lambda: handle_error_dialog(run_dialog))
        qtbot.waitUntil(run_dialog.done_button.isVisible, timeout=200000)
        run_dialog.close()


@pytest.mark.integration_test
@pytest.mark.usefixtures("use_tmpdir")
def test_that_debug_info_button_provides_data_in_clipboard(qtbot: QtBot, storage):
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
        experiment_panel = gui.findChild(ExperimentPanel)
        assert isinstance(experiment_panel, ExperimentPanel)

        run_experiment = experiment_panel.findChild(QWidget, name="run_experiment")
        assert run_experiment
        assert isinstance(run_experiment, QToolButton)

        qtbot.mouseClick(run_experiment, Qt.LeftButton)
        qtbot.waitUntil(lambda: gui.findChild(RunDialog) is not None, timeout=5000)
        run_dialog = gui.findChild(RunDialog)
        qtbot.waitUntil(run_dialog.done_button.isVisible, timeout=100000)

        copy_debug_info_button = gui.findChild(QPushButton, "copy_debug_info_button")
        assert copy_debug_info_button
        assert isinstance(copy_debug_info_button, QPushButton)
        qtbot.mouseClick(copy_debug_info_button, Qt.LeftButton)

        clipboard_text = QApplication.clipboard().text()

        for keyword in ["Single realization test-run", "Local", r"minimal\_config.ert"]:
            assert keyword in clipboard_text
        qtbot.mouseClick(run_dialog.done_button, Qt.LeftButton)


@pytest.mark.integration_test
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
        qtbot.waitUntil(run_dialog.done_button.isVisible, timeout=100000)
        fm_step_overview = run_dialog._fm_step_overview
        qtbot.waitUntil(fm_step_overview.isVisible, timeout=20000)

        realization_widget = run_dialog.findChild(RealizationWidget)

        click_pos = realization_widget._real_view.rectForIndex(
            realization_widget._real_list_model.index(0, 0)
        ).center()

        with qtbot.waitSignal(realization_widget.itemClicked, timeout=30000):
            qtbot.mouseClick(
                realization_widget._real_view.viewport(),
                Qt.LeftButton,
                pos=click_pos,
            )

        fm_step_stdout = fm_step_overview.model().index(0, 4)
        fm_step_stderr = fm_step_overview.model().index(0, 5)

        assert fm_step_stdout.data(Qt.ItemDataRole.DisplayRole) == "View"
        assert fm_step_stderr.data(Qt.ItemDataRole.DisplayRole) == "-"
        assert (
            fm_step_stdout.data(Qt.ItemDataRole.ForegroundRole) == Qt.GlobalColor.blue
        )
        assert fm_step_stderr.data(Qt.ItemDataRole.ForegroundRole) == None

        assert fm_step_stdout.data(Qt.ItemDataRole.FontRole).underline() == True
        assert fm_step_stderr.data(Qt.ItemDataRole.FontRole) == None

        click_pos = fm_step_overview.visualRect(fm_step_stdout).center()

        qtbot.mouseClick(fm_step_overview.viewport(), Qt.LeftButton, pos=click_pos)

        qtbot.waitUntil(run_dialog.findChild(FileDialog).isVisible, timeout=30000)

        with qtbot.waitSignal(run_dialog.accepted, timeout=30000):
            run_dialog.close()


@pytest.mark.integration_test
@pytest.mark.usefixtures("use_tmpdir")
@pytest.mark.parametrize(
    "design_matrix_entry",
    (True, False),
)
def test_that_design_matrix_show_parameters_button_is_visible(
    design_matrix_entry, qtbot: QtBot, storage
):
    xls_filename = "design_matrix.xls"
    with open(f"{xls_filename}", "w", encoding="utf-8"):
        pass
    config_file = "minimal_config.ert"
    with open(config_file, "w", encoding="utf-8") as f:
        f.write("NUM_REALIZATIONS 1")
        if design_matrix_entry:
            f.write(
                f"\nDESIGN_MATRIX {xls_filename} DESIGN_SHEET:DesignSheet01 DEFAULT_SHEET:DefaultValues"
            )

    args_mock = Mock()
    args_mock.config = config_file

    ert_config = ErtConfig.from_file(config_file)
    with StorageService.init_service(
        project=os.path.abspath(ert_config.ens_path),
    ):
        gui = _setup_main_window(ert_config, args_mock, GUILogHandler(), storage)
        experiment_panel = gui.findChild(ExperimentPanel)
        assert isinstance(experiment_panel, ExperimentPanel)

        simulation_mode_combo = experiment_panel.findChild(QComboBox)
        assert isinstance(simulation_mode_combo, QComboBox)

        simulation_mode_combo.setCurrentText(EnsembleExperiment.name())
        simulation_settings = gui.findChild(EnsembleExperimentPanel)
        show_dm_parameters = simulation_settings.findChild(
            QPushButton, "show-dm-parameters"
        )
        if design_matrix_entry:
            assert isinstance(show_dm_parameters, QPushButton)
        else:
            assert show_dm_parameters is None
