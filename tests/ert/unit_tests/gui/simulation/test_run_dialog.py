import tempfile
from pathlib import Path
from unittest.mock import MagicMock, Mock, patch

import pandas as pd
import pytest
from PyQt6.QtCore import Qt, QTimer
from PyQt6.QtWidgets import (
    QApplication,
    QComboBox,
    QDialogButtonBox,
    QLabel,
    QMessageBox,
    QPushButton,
    QToolButton,
    QWidget,
)
from pytestqt.qtbot import QtBot

import ert.run_models
from _ert.events import EnsembleEvaluationWarning
from ert.config import ErtConfig
from ert.ensemble_evaluator import state
from ert.ensemble_evaluator.event import (
    EndEvent,
    FullSnapshotEvent,
    SnapshotUpdateEvent,
)
from ert.gui.main import GUILogHandler, _setup_main_window
from ert.gui.simulation.ensemble_experiment_panel import (
    DesignMatrixPanel,
    EnsembleExperimentPanel,
)
from ert.gui.simulation.ensemble_smoother_panel import EnsembleSmootherPanel
from ert.gui.simulation.experiment_panel import ExperimentPanel
from ert.gui.simulation.multiple_data_assimilation_panel import (
    MultipleDataAssimilationPanel,
)
from ert.gui.simulation.run_dialog import RunDialog
from ert.gui.simulation.view.realization import RealizationWidget
from ert.gui.tools.file import FileDialog
from ert.run_models import (
    EnsembleExperiment,
    EnsembleSmoother,
    MultipleDataAssimilation,
)
from ert.scheduler.job import Job
from tests.ert import SnapshotBuilder
from tests.ert.ui_tests.gui.conftest import wait_for_child
from tests.ert.unit_tests.gui.simulation.test_run_path_dialog import (
    handle_run_path_dialog,
)
from tests.ert.utils import wait_until


@pytest.fixture
def event_queue(events):
    async def _add_event(self, *_):
        for event in events:
            self.send_event(event)
        return [0]

    with patch(
        "ert.run_models.run_model.RunModel.run_ensemble_evaluator_async",
        _add_event,
    ):
        yield


@pytest.fixture
def event_queue_large_snapshot(large_snapshot):
    events = [
        FullSnapshotEvent(
            snapshot=large_snapshot,
            iteration_label="Foo",
            total_iterations=1,
            progress=0.5,
            realization_count=4,
            status_count={"Finished": 2, "Unknown": 2},
            iteration=0,
        ),
        FullSnapshotEvent(
            snapshot=large_snapshot,
            iteration_label="Foo",
            total_iterations=1,
            progress=0.5,
            realization_count=4,
            status_count={"Finished": 2, "Unknown": 2},
            iteration=1,
        ),
        EndEvent(failed=False, msg=""),
    ]

    async def _add_event(self, *_):
        for event in events:
            self.send_event(event)
        return [0]

    with patch(
        "ert.run_models.run_model.RunModel.run_ensemble_evaluator_async",
        _add_event,
    ):
        yield


@pytest.fixture
def mock_set_is_simulation_running():
    mock = MagicMock()
    with patch(
        "ert.gui.main_window.ErtNotifier.set_is_simulation_running", mock
    ) as _mock:
        yield _mock


@pytest.fixture
def mock_set_env_key():
    mock = MagicMock()
    with patch("ert.run_models.run_model.RunModel.set_env_key", mock) as _mock:
        yield _mock


@pytest.fixture
def run_dialog(qtbot: QtBot, use_tmpdir, mock_set_env_key, monkeypatch):
    config_file = "minimal_config.ert"
    monkeypatch.setattr("ert.scheduler.Scheduler.BATCH_KILLING_INTERVAL", 0.01)
    monkeypatch.setattr(
        "ert.ensemble_evaluator.EnsembleEvaluator.BATCHING_INTERVAL", 0.01
    )
    Path(config_file).write_text(
        "NUM_REALIZATIONS 1\nQUEUE_SYSTEM LOCAL", encoding="utf-8"
    )
    args_mock = Mock()
    args_mock.config = config_file
    ert_config = ErtConfig.from_file(config_file)
    gui = _setup_main_window(ert_config, args_mock, GUILogHandler(), "storage")
    qtbot.addWidget(gui)
    experiment_panel = gui.findChild(ExperimentPanel)
    assert experiment_panel
    simulation_mode_combo = experiment_panel.findChild(QComboBox)
    assert simulation_mode_combo
    simulation_mode_combo.setCurrentText(EnsembleExperiment.name())
    simulation_settings = gui.findChild(EnsembleExperimentPanel)
    simulation_settings._experiment_name_field.setText("new_experiment_name")
    run_experiment = experiment_panel.findChild(QToolButton, name="run_experiment")
    assert run_experiment
    qtbot.mouseClick(run_experiment, Qt.MouseButton.LeftButton)
    qtbot.waitUntil(lambda: gui.findChild(RunDialog) is not None, timeout=5000)
    run_dialog = gui.findChild(RunDialog)
    assert run_dialog
    yield run_dialog


@pytest.mark.integration_test
def test_that_terminating_experiment_shows_a_confirmation_dialog(
    qtbot: QtBot, run_dialog: RunDialog, monkeypatch
):
    monkeypatch.setattr(Job, "WAIT_PERIOD_FOR_TERM_MESSAGE_TO_CANCEL", 0)
    kill_button = run_dialog.kill_button
    with qtbot.waitSignal(run_dialog.simulation_done, timeout=10000):

        def handle_dialog():
            terminate_dialog = wait_for_child(run_dialog, qtbot, QMessageBox)
            dialog_buttons = terminate_dialog.findChild(QDialogButtonBox).buttons()
            yes_button = next(b for b in dialog_buttons if "Yes" in b.text())
            qtbot.mouseClick(yes_button, Qt.MouseButton.LeftButton)

        QTimer.singleShot(100, handle_dialog)
        assert kill_button.isEnabled()
        assert kill_button.text() == "Terminate experiment"
        qtbot.mouseClick(run_dialog.kill_button, Qt.MouseButton.LeftButton)
        assert not kill_button.isEnabled()
        assert kill_button.text() == "Terminating"
    wait_until(lambda: run_dialog.fail_msg_box is not None, timeout=5000)
    assert kill_button.text() == "Terminate experiment"
    assert not kill_button.isEnabled()
    assert (
        "Experiment cancelled by user during evaluation"
        in run_dialog.fail_msg_box.findChild(QWidget, name="suggestor_messages")
        .findChild(QLabel)
        .text()
    )


@pytest.mark.integration_test
def test_large_snapshot(
    event_queue_large_snapshot,
    qtbot: QtBot,
    run_dialog: RunDialog,
    timeout_per_iter=5000,
):
    qtbot.waitUntil(
        lambda: run_dialog._total_progress_bar.value() == 100,
        timeout=timeout_per_iter * 3,
    )
    qtbot.waitUntil(
        lambda: run_dialog._tab_widget.count() == 2, timeout=timeout_per_iter
    )
    qtbot.waitUntil(
        lambda: run_dialog.is_simulation_done() is True, timeout=timeout_per_iter
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
def test_run_dialog(events, event_queue, tab_widget_count, qtbot: QtBot, run_dialog):
    qtbot.waitUntil(
        lambda: run_dialog._tab_widget.count() == tab_widget_count, timeout=5000
    )
    qtbot.waitUntil(lambda: run_dialog.is_simulation_done() is True, timeout=5000)


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
    events, event_queue, tab_widget_count, qtbot: QtBot, run_dialog
):
    qtbot.waitUntil(
        lambda: run_dialog._tab_widget.count() == tab_widget_count, timeout=5000
    )
    qtbot.waitUntil(lambda: run_dialog.is_simulation_done() is True, timeout=5000)

    # This is the container of realization boxes
    realization_box = run_dialog._tab_widget.widget(0)
    assert type(realization_box) is RealizationWidget
    # Click the first realization box
    qtbot.mouseClick(realization_box, Qt.MouseButton.LeftButton)
    fm_step_model = run_dialog._fm_step_overview.model()
    assert fm_step_model._real == 0

    fm_step_number = 0
    max_memory_column_index = 6

    max_memory_column_proxy_index = fm_step_model.index(
        fm_step_number, max_memory_column_index
    )
    max_memory_value = fm_step_model.data(
        max_memory_column_proxy_index, Qt.ItemDataRole.DisplayRole
    )
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
                        .add_fm_step(
                            fm_step_id="1",
                            index="1",
                            name="fm_step_1",
                            status=state.FORWARD_MODEL_STATE_START,
                        )
                        .build(
                            ["0", "1"],
                            status=state.REALIZATION_STATE_UNKNOWN,
                            exec_hosts="COMP_01",
                        )
                    ),
                    iteration_label="Foo",
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
                        .add_fm_step(
                            fm_step_id="1",
                            index="1",
                            name="fm_step_1",
                            status=state.FORWARD_MODEL_STATE_START,
                        )
                        .build(["0", "1"], status=state.REALIZATION_STATE_UNKNOWN)
                    ),
                    iteration_label="Foo",
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
    events, event_queue, tab_widget_count, expected_host_info, qtbot: QtBot, run_dialog
):
    qtbot.waitUntil(
        lambda: run_dialog._tab_widget.count() == tab_widget_count, timeout=5000
    )
    qtbot.waitUntil(lambda: run_dialog.is_simulation_done() is True, timeout=5000)

    # This is the container of realization boxes
    realization_box = run_dialog._tab_widget.widget(0)
    assert type(realization_box) is RealizationWidget
    # Click the first realization box
    qtbot.mouseClick(realization_box, Qt.MouseButton.LeftButton)
    fm_step_model = run_dialog._fm_step_overview.model()
    assert fm_step_model._real == 0

    # default selection should yield realization 0, iteration 0
    fm_step_label = run_dialog.findChild(QLabel, name="fm_step_label")
    assert "Realization id 0 in iteration 0" in fm_step_label.text()

    # clicking realization 1 should update fm_label with realization 1, iteration 0
    realization_box._item_clicked(run_dialog._fm_step_overview.model().index(1, 0))

    assert (
        fm_step_label.text() == f"Realization id 1 in iteration 0{expected_host_info}"
    )


@pytest.mark.integration_test
@pytest.mark.usefixtures("use_tmpdir")
def test_that_exception_in_run_model_is_displayed_in_a_suggestor_window_after_simulation_fails(  # noqa E501
    qtbot: QtBot, use_tmpdir
):
    config_file = "minimal_config.ert"
    Path(config_file).write_text(
        "NUM_REALIZATIONS 1\nQUEUE_SYSTEM LOCAL", encoding="utf-8"
    )
    args_mock = Mock()
    args_mock.config = config_file

    ert_config = ErtConfig.from_file(config_file)
    with patch.object(
        ert.run_models.SingleTestRun,
        "run_experiment",
        MagicMock(side_effect=ValueError("I failed :(")),
    ):
        gui = _setup_main_window(ert_config, args_mock, GUILogHandler(), "storage")
        qtbot.addWidget(gui)
        run_experiment = gui.findChild(QToolButton, name="run_experiment")

        handler_done = False

        def assert_failure_in_error_dialog(run_dialog):
            nonlocal handler_done
            wait_until(lambda: run_dialog.fail_msg_box is not None, timeout=10000)
            suggestor_termination_window = run_dialog.fail_msg_box
            assert suggestor_termination_window
            text = (
                suggestor_termination_window.findChild(
                    QWidget, name="suggestor_messages"
                )
                .findChild(QLabel)
                .text()
            )
            assert "I failed :(" in text
            button = suggestor_termination_window.findChild(
                QPushButton, name="close_button"
            )
            assert button
            button.click()
            handler_done = True

        simulation_mode_combo = gui.findChild(QComboBox)
        simulation_mode_combo.setCurrentText("Single realization test-run")
        qtbot.mouseClick(run_experiment, Qt.MouseButton.LeftButton)
        run_dialog = wait_for_child(gui, qtbot, RunDialog)

        QTimer.singleShot(100, lambda: assert_failure_in_error_dialog(run_dialog))
        # Capturing exceptions in order to catch an assertion error
        # from assert_failure_in_error_dialog and stop waiting
        with qtbot.captureExceptions() as exceptions:
            qtbot.waitUntil(
                lambda: run_dialog.is_simulation_done() is True or bool(exceptions),
                timeout=100000,
            )
            qtbot.waitUntil(lambda: handler_done or bool(exceptions), timeout=100000)
        if exceptions:
            raise AssertionError(
                f"Exception(s) happened in Qt event loop: {exceptions}"
            )


@pytest.mark.integration_test
def test_that_stdout_and_stderr_buttons_react_to_file_content(
    snake_oil_case_storage: ErtConfig, qtbot: QtBot
):
    snake_oil_case = snake_oil_case_storage
    args_mock = Mock()
    args_mock.config = "snake_oil.ert"

    gui = _setup_main_window(
        snake_oil_case, args_mock, GUILogHandler(), snake_oil_case_storage.ens_path
    )
    experiment_panel = gui.findChild(ExperimentPanel)
    assert experiment_panel
    simulation_mode_combo = experiment_panel.findChild(QComboBox)
    assert simulation_mode_combo
    simulation_mode_combo.setCurrentText(EnsembleExperiment.name())
    simulation_settings = gui.findChild(EnsembleExperimentPanel)
    simulation_settings._experiment_name_field.setText("new_experiment_name")

    run_experiment = experiment_panel.findChild(QToolButton, name="run_experiment")
    assert run_experiment

    QTimer.singleShot(
        1000, lambda: handle_run_path_dialog(gui, qtbot, delete_run_path=True)
    )
    qtbot.mouseClick(run_experiment, Qt.MouseButton.LeftButton)
    qtbot.waitUntil(lambda: gui.findChild(RunDialog) is not None, timeout=5000)
    run_dialog = gui.findChild(RunDialog)

    qtbot.waitUntil(lambda: run_dialog.is_simulation_done() is True, timeout=100000)

    fm_step_overview = run_dialog._fm_step_overview
    qtbot.waitUntil(lambda: not fm_step_overview.isHidden(), timeout=20000)
    realization_widget = run_dialog.findChild(RealizationWidget)

    click_pos = realization_widget._real_view.rectForIndex(
        realization_widget._real_list_model.index(0, 0)
    ).center()

    with qtbot.waitSignal(realization_widget.itemClicked, timeout=30000):
        qtbot.mouseClick(
            realization_widget._real_view.viewport(),
            Qt.MouseButton.LeftButton,
            pos=click_pos,
        )

    fm_step_stdout = fm_step_overview.model().index(0, 4)
    fm_step_stderr = fm_step_overview.model().index(0, 5)

    assert fm_step_stdout.data(Qt.ItemDataRole.DisplayRole) == "View"
    assert fm_step_stderr.data(Qt.ItemDataRole.DisplayRole) == "-"
    assert fm_step_stdout.data(Qt.ItemDataRole.ForegroundRole) == Qt.GlobalColor.blue
    assert fm_step_stderr.data(Qt.ItemDataRole.ForegroundRole) is None
    assert fm_step_stdout.data(Qt.ItemDataRole.FontRole).underline()
    assert fm_step_stderr.data(Qt.ItemDataRole.FontRole) is None

    click_pos = fm_step_overview.visualRect(fm_step_stdout).center()
    qtbot.mouseClick(
        fm_step_overview.viewport(), Qt.MouseButton.LeftButton, pos=click_pos
    )
    file_dialog = run_dialog.findChild(FileDialog)
    qtbot.waitUntil(file_dialog.isVisible, timeout=10000)
    file_dialog.close()


@pytest.mark.integration_test
@pytest.mark.usefixtures("use_tmpdir")
@pytest.mark.parametrize(
    "design_matrix_entry",
    (True, False),
)
@pytest.mark.filterwarnings("ignore:NUM_REALIZATIONS")
def test_that_design_matrix_show_parameters_button_is_visible(
    design_matrix_entry, qtbot: QtBot, use_tmpdir
):
    xls_filename = "design_matrix.xlsx"
    design_matrix_df = pd.DataFrame(
        {
            "REAL": [0, 10, 20],
            "a": [0, 1, 5],
        }
    )
    with pd.ExcelWriter(xls_filename) as xl_write:
        design_matrix_df.to_excel(xl_write, index=False, sheet_name="DesignSheet")

    config_file = "minimal_config.ert"
    design_matrix_config = (
        f"\nDESIGN_MATRIX {xls_filename}" if design_matrix_entry else ""
    )
    Path(config_file).write_text(
        "NUM_REALIZATIONS 1\nQUEUE_SYSTEM LOCAL" + design_matrix_config,
        encoding="utf-8",
    )

    args_mock = Mock()
    args_mock.config = config_file

    ert_config = ErtConfig.from_file(config_file)
    gui = _setup_main_window(ert_config, args_mock, GUILogHandler(), "storage")
    experiment_panel = gui.findChild(ExperimentPanel)
    assert experiment_panel

    simulation_mode_combo = experiment_panel.findChild(QComboBox)
    assert simulation_mode_combo

    simulation_mode_combo.setCurrentText(EnsembleExperiment.name())
    simulation_settings = gui.findChild(EnsembleExperimentPanel)
    show_dm_parameters = simulation_settings.findChild(
        QPushButton, "show-dm-parameters"
    )
    if design_matrix_entry:
        assert show_dm_parameters

        def dialog_appeared_and_test():
            app = QApplication.instance()
            dialogs = [w for w in app.allWidgets() if isinstance(w, DesignMatrixPanel)]

            if not dialogs:
                raise AssertionError("No DesignMatrixPanel dialog found")
            dialog = dialogs[0]

            try:
                model = dialog.model
                assert model.rowCount() == 3
                assert model.columnCount() == 1
                assert model.horizontalHeaderItem(0).text() == "a"

                assert model.verticalHeaderItem(0).text() == "0"
                assert model.verticalHeaderItem(1).text() == "10"
                assert model.verticalHeaderItem(2).text() == "20"

                assert model.item(0, 0).text() == "0"
                assert model.item(1, 0).text() == "1"
                assert model.item(2, 0).text() == "5"

                dialog.accept()
            except Exception as e:
                dialog.accept()
                raise e

        QTimer.singleShot(500, dialog_appeared_and_test)
        qtbot.mouseClick(show_dm_parameters, Qt.MouseButton.LeftButton)
    else:
        assert show_dm_parameters is None


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
                        .build(["0", "1"], state.REALIZATION_STATE_UNKNOWN)
                    ),
                    iteration_label="Foo",
                    total_iterations=1,
                    progress=0.5,
                    realization_count=2,
                    status_count={"Finished": 1, "Pending": 1},
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
                        .build(["0", "1"], state.REALIZATION_STATE_FINISHED)
                    ),
                    iteration_label="Foo",
                    total_iterations=1,
                    progress=0.5,
                    realization_count=2,
                    status_count={"Finished": 1, "Pending": 1},
                    iteration=1,
                ),
                EndEvent(failed=False, msg=""),
            ],
            2,
            id="changing from between tabs",
        ),
    ],
)
def test_forward_model_overview_label_selected_on_tab_change(
    events, event_queue, tab_widget_count, qtbot: QtBot, run_dialog
):
    def qt_bot_click_realization(realization_index: int, iteration: int) -> None:
        view = run_dialog._tab_widget.widget(iteration)._real_view
        model_index = view.model().index(realization_index, 0)
        view.scrollTo(model_index)
        rect = view.visualRect(model_index)
        click_pos = rect.center()
        qtbot.mouseClick(view.viewport(), Qt.MouseButton.LeftButton, pos=click_pos)

    def qt_bot_click_tab_index(tab_index: int) -> None:
        tab_bar = run_dialog._tab_widget.tabBar()
        tab_rect = tab_bar.tabRect(tab_index)
        click_pos = tab_rect.center()
        qtbot.mouseClick(tab_bar, Qt.MouseButton.LeftButton, pos=click_pos)

    # verify two tabs present
    qtbot.waitUntil(
        lambda: run_dialog._tab_widget.count() == tab_widget_count, timeout=5000
    )
    qtbot.waitUntil(lambda: run_dialog.is_simulation_done() is True, timeout=5000)

    qt_bot_click_tab_index(0)
    fm_step_label = run_dialog.findChild(QLabel, name="fm_step_label")
    assert "Realization id 0 in iteration 0" in fm_step_label.text()

    qt_bot_click_realization(1, 0)
    assert "Realization id 1 in iteration 0" in fm_step_label.text()

    qt_bot_click_tab_index(1)
    assert "Realization id 0 in iteration 1" in fm_step_label.text()

    qt_bot_click_realization(1, 1)
    assert "Realization id 1 in iteration 1" in fm_step_label.text()

    qt_bot_click_tab_index(0)
    assert "Realization id 1 in iteration 0" in fm_step_label.text()

    qt_bot_click_tab_index(1)
    assert "Realization id 1 in iteration 1" in fm_step_label.text()


@pytest.mark.integration_test
@pytest.mark.usefixtures("use_tmpdir")
@pytest.mark.parametrize(
    "experiment_mode, experiment_mode_panel, dm_realizations",
    [
        (EnsembleExperiment, EnsembleExperimentPanel, 5),
        (EnsembleExperiment, EnsembleExperimentPanel, 15),
        (EnsembleSmoother, EnsembleSmootherPanel, 5),
        (EnsembleSmoother, EnsembleSmootherPanel, 15),
        (MultipleDataAssimilation, MultipleDataAssimilationPanel, 5),
        (MultipleDataAssimilation, MultipleDataAssimilationPanel, 15),
    ],
)
@pytest.mark.filterwarnings("ignore:NUM_REALIZATIONS")
def test_that_ert_chooses_minimum_realization_with_design_matrix(
    qtbot: QtBot, experiment_mode, dm_realizations, experiment_mode_panel, use_tmpdir
):
    xls_filename = "design_matrix.xlsx"
    design_matrix_df = pd.DataFrame(
        {
            "REAL": list(range(dm_realizations)),
            "a": list(range(dm_realizations)),
        }
    )
    default_sheet_df = pd.DataFrame([["b", 1], ["c", 2]])
    with pd.ExcelWriter(xls_filename) as xl_write:
        design_matrix_df.to_excel(xl_write, index=False, sheet_name="DesignSheet")
        default_sheet_df.to_excel(
            xl_write, index=False, sheet_name="DefaultSheet", header=False
        )

    config_num_realizations = 10
    expected_num_realizations = min(dm_realizations, config_num_realizations)
    config_file = "minimal_config.ert"
    Path(config_file).write_text(
        (
            f"NUM_REALIZATIONS {config_num_realizations}\n"
            f"DESIGN_MATRIX {xls_filename}\n"
            "QUEUE_SYSTEM LOCAL"
        ),
        encoding="utf-8",
    )

    args_mock = Mock()
    args_mock.config = config_file

    ert_config = ErtConfig.from_file(config_file)
    gui = _setup_main_window(ert_config, args_mock, GUILogHandler(), "storage")
    experiment_panel = gui.findChild(ExperimentPanel)
    assert experiment_panel

    simulation_mode_combo = experiment_panel.findChild(QComboBox)
    assert simulation_mode_combo

    simulation_mode_combo.setCurrentText(experiment_mode.name())
    simulation_settings = gui.findChild(experiment_mode_panel)
    num_realizations_label = simulation_settings.findChild(QWidget, "num_reals_label")
    assert num_realizations_label

    assert num_realizations_label.text() == f"<b>{expected_num_realizations}</b>"

    # Verify that the warning icon has the correct tooltip
    warning_icon = gui.findChild(QLabel, "warning_icon_num_realizations_design_matrix")
    assert warning_icon.toolTip() == (
        f"Number of realizations was set to {expected_num_realizations} "
        "due to different number of realizations in the design matrix "
        "and NUM_REALIZATIONS in config"
    )


@pytest.mark.integration_test
def test_that_file_dialog_close_when_run_dialog_hidden(qtbot: QtBot, run_dialog):
    with qtbot.waitSignal(run_dialog.simulation_done, timeout=10000):
        assert not run_dialog.findChild(FileDialog)  # No file dialog from fixture setup

        with tempfile.NamedTemporaryFile(mode="w", encoding="utf-8") as tmp_file:
            FileDialog._init_thread = (
                lambda self: None
            )  # To avoid firing up the file watcher
            file_dialog = FileDialog(tmp_file.name, "the_step", 0, 0, 0, run_dialog)
            assert run_dialog.findChild(FileDialog)
            assert file_dialog.isVisible()
            run_dialog.setVisible(False)
            assert not file_dialog.isVisible()


def test_that_run_dialog_clears_warnings_when_rerun(qtbot, monkeypatch):
    run_dialog = RunDialog(
        title="test",
        run_model_api=MagicMock(),
        event_queue=MagicMock(),
        notifier=MagicMock(),
    )
    assert len(run_dialog.post_simulation_warnings) == 0
    run_dialog.post_simulation_warnings.append("warning")
    assert len(run_dialog.post_simulation_warnings) > 0

    monkeypatch.setattr(
        QMessageBox, "exec", value=lambda _: QMessageBox.StandardButton.Ok
    )
    run_dialog.rerun_failed_realizations()
    assert len(run_dialog.post_simulation_warnings) == 0


@pytest.mark.integration_test
@pytest.mark.parametrize(
    "events",
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
                        .build(["0", "1"], state.REALIZATION_STATE_UNKNOWN)
                    ),
                    iteration_label="Foo",
                    total_iterations=1,
                    progress=0.5,
                    realization_count=2,
                    status_count={"Finished": 1, "Pending": 1},
                    iteration=0,
                ),
                EnsembleEvaluationWarning(warning_message="foo_bar_error"),
                FullSnapshotEvent(
                    snapshot=(
                        SnapshotBuilder()
                        .add_fm_step(
                            fm_step_id="0",
                            index="0",
                            name="fm_step_0",
                            status=state.FORWARD_MODEL_STATE_START,
                        )
                        .build(["0", "1"], state.REALIZATION_STATE_FINISHED)
                    ),
                    iteration_label="Foo",
                    total_iterations=1,
                    progress=0.5,
                    realization_count=2,
                    status_count={"Finished": 1, "Pending": 1},
                    iteration=1,
                ),
                EndEvent(failed=False, msg=""),
            ],
            id="scheduler_warning_event_between_snapshot_events",
        ),
    ],
)
def test_that_experiment_with_a_scheduler_warning_event_shows_a_warning_dialog(
    events, event_queue, qtbot: QtBot, run_dialog: RunDialog
):
    with qtbot.waitSignal(run_dialog.simulation_done, timeout=10000):

        def handle_dialog():
            ensemble_evaluation_warning_box = wait_for_child(
                run_dialog, qtbot, QMessageBox
            )

            assert ensemble_evaluation_warning_box.text() == "foo_bar_error"

            dialog_buttons = wait_for_child(
                ensemble_evaluation_warning_box, qtbot, QDialogButtonBox
            ).buttons()
            assert (
                ensemble_evaluation_warning_box.objectName()
                == "EnsembleEvaluationWarningBox"
            )
            yes_button = next(b for b in dialog_buttons if "OK" in b.text())
            qtbot.mouseClick(yes_button, Qt.MouseButton.LeftButton)

        handle_dialog()

    assert run_dialog is not None
