from unittest.mock import MagicMock, Mock, call, patch

import pandas as pd
import pytest
from PyQt6.QtCore import Qt, QTimer
from PyQt6.QtWidgets import (
    QComboBox,
    QDialogButtonBox,
    QLabel,
    QMessageBox,
    QPushButton,
    QToolButton,
    QWidget,
)
from pytestqt.qtbot import QtBot

import ert
import ert.run_models
from ert.config import ErtConfig
from ert.ensemble_evaluator import state
from ert.ensemble_evaluator.event import (
    EndEvent,
    FullSnapshotEvent,
    SnapshotUpdateEvent,
)
from ert.gui.ertwidgets.message_box import ErtMessageBox
from ert.gui.main import GUILogHandler, _setup_main_window
from ert.gui.simulation.ensemble_experiment_panel import EnsembleExperimentPanel
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
from ert.storage import open_storage
from tests.ert import SnapshotBuilder
from tests.ert.ui_tests.gui.conftest import wait_for_child
from tests.ert.unit_tests.gui.simulation.test_run_path_dialog import (
    handle_run_path_dialog,
)


@pytest.fixture
def event_queue(events):
    async def _add_event(self, *_):
        for event in events:
            self.send_event(event)
        return [0]

    with patch(
        "ert.run_models.base_run_model.BaseRunModel.run_ensemble_evaluator_async",
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
        "ert.run_models.base_run_model.BaseRunModel.run_ensemble_evaluator_async",
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
def mock_get_runtime():
    mock = MagicMock()
    with patch("ert.run_models.base_run_model.BaseRunModel.get_runtime", mock) as _mock:
        _mock.return_value = 10
        yield _mock


@pytest.fixture
def mock_set_env_key():
    mock = MagicMock()
    with patch("ert.run_models.base_run_model.BaseRunModel.set_env_key", mock) as _mock:
        yield _mock


@pytest.fixture
def run_dialog(qtbot: QtBot, use_tmpdir, storage, mock_set_env_key):
    config_file = "minimal_config.ert"
    with open(config_file, "w", encoding="utf-8") as f:
        f.write("NUM_REALIZATIONS 1")
    args_mock = Mock()
    args_mock.config = config_file
    ert_config = ErtConfig.from_file(config_file)
    gui = _setup_main_window(ert_config, args_mock, GUILogHandler(), storage)
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
def test_terminating_experiment_shows_a_confirmation_dialog(qtbot: QtBot, run_dialog):
    with qtbot.waitSignal(run_dialog.simulation_done, timeout=10000):

        def handle_dialog():
            terminate_dialog = wait_for_child(run_dialog, qtbot, QMessageBox)
            dialog_buttons = terminate_dialog.findChild(QDialogButtonBox).buttons()
            yes_button = next(b for b in dialog_buttons if "Yes" in b.text())
            qtbot.mouseClick(yes_button, Qt.MouseButton.LeftButton)

        QTimer.singleShot(100, handle_dialog)
        qtbot.mouseClick(run_dialog.kill_button, Qt.MouseButton.LeftButton)
    terminate_info = wait_for_child(run_dialog, qtbot, ErtMessageBox)
    assert (
        terminate_info.details_text.toPlainText()
        == "Experiment cancelled by user during evaluation\n"
    )


@pytest.mark.integration_test
def test_run_dialog_polls_run_model_for_runtime(
    qtbot, mock_set_is_simulation_running, mock_get_runtime, run_dialog: RunDialog
):
    qtbot.waitUntil(lambda: run_dialog.is_simulation_done() is True)
    mock_get_runtime.assert_any_call()
    mock_set_is_simulation_running.assert_has_calls([call(True), call(False)])


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
def test_that_exception_in_base_run_model_is_handled(qtbot: QtBot, storage):
    config_file = "minimal_config.ert"
    with open(config_file, "w", encoding="utf-8") as f:
        f.write("NUM_REALIZATIONS 1")
    args_mock = Mock()
    args_mock.config = config_file

    ert_config = ErtConfig.from_file(config_file)
    with patch.object(
        ert.run_models.SingleTestRun,
        "run_experiment",
        MagicMock(side_effect=ValueError("I failed :(")),
    ):
        gui = _setup_main_window(ert_config, args_mock, GUILogHandler(), storage)
        qtbot.addWidget(gui)
        run_experiment = gui.findChild(QToolButton, name="run_experiment")

        handler_done = False

        def handle_error_dialog(run_dialog):
            nonlocal handler_done
            qtbot.waitUntil(
                lambda: run_dialog.fail_msg_box is not None,
                timeout=20000,
            )
            error_dialog = run_dialog.fail_msg_box
            assert error_dialog
            text = error_dialog.details_text.toPlainText()
            assert "I failed :(" in text
            qtbot.mouseClick(error_dialog.box.buttons()[0], Qt.MouseButton.LeftButton)
            handler_done = True

        simulation_mode_combo = gui.findChild(QComboBox)
        simulation_mode_combo.setCurrentText("Single realization test-run")
        qtbot.mouseClick(run_experiment, Qt.MouseButton.LeftButton)
        run_dialog = wait_for_child(gui, qtbot, RunDialog)

        QTimer.singleShot(100, lambda: handle_error_dialog(run_dialog))
        qtbot.waitUntil(lambda: run_dialog.is_simulation_done() is True, timeout=100000)
        qtbot.waitUntil(lambda: handler_done, timeout=100000)


@pytest.mark.integration_test
def test_that_stdout_and_stderr_buttons_react_to_file_content(
    snake_oil_case_storage: ErtConfig, qtbot: QtBot
):
    snake_oil_case = snake_oil_case_storage
    args_mock = Mock()
    args_mock.config = "snake_oil.ert"

    with (
        open_storage(snake_oil_case.ens_path, mode="w") as storage,
    ):
        gui = _setup_main_window(snake_oil_case, args_mock, GUILogHandler(), storage)
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
        assert (
            fm_step_stdout.data(Qt.ItemDataRole.ForegroundRole) == Qt.GlobalColor.blue
        )
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
def test_that_design_matrix_show_parameters_button_is_visible(
    design_matrix_entry, qtbot: QtBot, storage
):
    xls_filename = "design_matrix.xlsx"
    design_matrix_df = pd.DataFrame(
        {
            "REAL": list(range(3)),
            "a": [0, 1, 2],
        }
    )
    default_sheet_df = pd.DataFrame([["b", 1], ["c", 2]])
    with pd.ExcelWriter(xls_filename) as xl_write:
        design_matrix_df.to_excel(xl_write, index=False, sheet_name="DesignSheet")
        default_sheet_df.to_excel(
            xl_write, index=False, sheet_name="DefaultSheet", header=False
        )

    config_file = "minimal_config.ert"
    with open(config_file, "w", encoding="utf-8") as f:
        f.write("NUM_REALIZATIONS 1")
        if design_matrix_entry:
            f.write(f"\nDESIGN_MATRIX {xls_filename}")

    args_mock = Mock()
    args_mock.config = config_file

    ert_config = ErtConfig.from_file(config_file)
    gui = _setup_main_window(ert_config, args_mock, GUILogHandler(), storage)
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
    "experiment_mode, experiment_mode_panel",
    [
        (EnsembleExperiment, EnsembleExperimentPanel),
        (EnsembleSmoother, EnsembleSmootherPanel),
        (MultipleDataAssimilation, MultipleDataAssimilationPanel),
    ],
)
def test_that_design_matrix_alters_num_realizations_field(
    qtbot: QtBot, storage, experiment_mode, experiment_mode_panel
):
    xls_filename = "design_matrix.xlsx"
    design_matrix_df = pd.DataFrame(
        {
            "REAL": list(range(3)),
            "a": [0, 1, 2],
        }
    )
    default_sheet_df = pd.DataFrame([["b", 1], ["c", 2]])
    with pd.ExcelWriter(xls_filename) as xl_write:
        design_matrix_df.to_excel(xl_write, index=False, sheet_name="DesignSheet")
        default_sheet_df.to_excel(
            xl_write, index=False, sheet_name="DefaultSheet", header=False
        )

    config_file = "minimal_config.ert"
    with open(config_file, "w", encoding="utf-8") as f:
        f.write("NUM_REALIZATIONS 10")
        f.write(f"\nDESIGN_MATRIX {xls_filename}")

    args_mock = Mock()
    args_mock.config = config_file

    ert_config = ErtConfig.from_file(config_file)
    gui = _setup_main_window(ert_config, args_mock, GUILogHandler(), storage)
    experiment_panel = gui.findChild(ExperimentPanel)
    assert experiment_panel

    simulation_mode_combo = experiment_panel.findChild(QComboBox)
    assert simulation_mode_combo

    simulation_mode_combo.setCurrentText(experiment_mode.name())
    simulation_settings = gui.findChild(experiment_mode_panel)
    num_realizations_label = simulation_settings.findChild(QWidget, "num_reals_label")
    assert num_realizations_label
    assert num_realizations_label.text() == "<b>3</b>"

    # Verify that the warning icon has the correct tooltip
    warning_icon = gui.findChild(QLabel, "warning_icon_num_realizations_design_matrix")
    assert warning_icon.toolTip() == (
        "Number of realizations changed from 10 to 3 due "
        "to 'REAL' column in design matrix"
    )
