import os
from pathlib import Path
from unittest.mock import Mock, patch

import pytest
from pytestqt.qtbot import QtBot
from qtpy import QtWidgets
from qtpy.QtCore import Qt, QTimer
from qtpy.QtWidgets import QMessageBox, QToolButton

from ert.config import ErtConfig
from ert.enkf_main import EnKFMain
from ert.ensemble_evaluator import state
from ert.ensemble_evaluator.event import (
    EndEvent,
    FullSnapshotEvent,
    SnapshotUpdateEvent,
)
from ert.ensemble_evaluator.snapshot import PartialSnapshot, SnapshotBuilder
from ert.gui.main import GUILogHandler, _setup_main_window
from ert.gui.simulation.run_dialog import RunDialog
from ert.gui.simulation.view.realization import RealizationWidget
from ert.gui.tools.file import FileDialog
from ert.services import StorageService


def test_success(runmodel, qtbot: QtBot, mock_tracker):
    notifier = Mock()
    widget = RunDialog("poly.ert", runmodel, notifier)
    widget.show()
    qtbot.addWidget(widget)

    with patch("ert.gui.simulation.run_dialog.EvaluatorTracker") as tracker:
        tracker.return_value = mock_tracker([EndEvent(failed=False, failed_msg="")])
        widget.startSimulation()

    with qtbot.waitExposed(widget, timeout=30000):
        qtbot.waitUntil(lambda: widget._total_progress_bar.value() == 100)
        qtbot.waitUntil(widget.done_button.isVisible, timeout=100)
        assert widget.done_button.text() == "Done"


# pylint: disable=no-member
def test_kill_simulations(runmodel, qtbot: QtBot, mock_tracker):
    notifier = Mock()
    widget = RunDialog("poly.ert", runmodel, notifier)
    widget.show()
    qtbot.addWidget(widget)

    with patch("ert.gui.simulation.run_dialog.EvaluatorTracker") as tracker:
        tracker.return_value = mock_tracker([EndEvent(failed=False, failed_msg="")])
        widget.startSimulation()

    with qtbot.waitSignal(widget.rejected, timeout=30000):

        def handle_dialog():
            qtbot.waitUntil(
                lambda: len(
                    [
                        x
                        for x in widget.children()
                        if isinstance(x, QtWidgets.QMessageBox)
                    ]
                )
                > 0
            )
            message_box = [
                x for x in widget.children() if isinstance(x, QtWidgets.QMessageBox)
            ][0]
            dialog_button_box = [
                x
                for x in message_box.children()
                if isinstance(x, QtWidgets.QDialogButtonBox)
            ][0]
            qtbot.mouseClick(dialog_button_box.children()[-2], Qt.LeftButton)

        QTimer.singleShot(100, handle_dialog)
        widget.killJobs()


def test_large_snapshot(
    runmodel, large_snapshot, qtbot: QtBot, mock_tracker, timeout_per_iter=5000
):
    notifier = Mock()
    widget = RunDialog("poly.ert", runmodel, notifier)
    widget.show()
    qtbot.addWidget(widget)

    with patch("ert.gui.simulation.run_dialog.EvaluatorTracker") as tracker:
        iter_0 = FullSnapshotEvent(
            snapshot=large_snapshot,
            phase_name="Foo",
            current_phase=0,
            total_phases=1,
            progress=0.5,
            iteration=0,
            indeterminate=False,
        )
        iter_1 = FullSnapshotEvent(
            snapshot=large_snapshot,
            phase_name="Foo",
            current_phase=0,
            total_phases=1,
            progress=0.5,
            iteration=1,
            indeterminate=False,
        )
        tracker.return_value = mock_tracker(
            [iter_0, iter_1, EndEvent(failed=False, failed_msg="")]
        )
        widget.startSimulation()

    with qtbot.waitExposed(widget, timeout=timeout_per_iter * 6):
        qtbot.waitUntil(
            lambda: widget._total_progress_bar.value() == 100,
            timeout=timeout_per_iter * 3,
        )
        qtbot.mouseClick(widget.show_details_button, Qt.LeftButton)
        qtbot.waitUntil(
            lambda: widget._tab_widget.count() == 2, timeout=timeout_per_iter
        )
        qtbot.waitUntil(
            lambda: widget.done_button.isVisible(), timeout=timeout_per_iter
        )


@pytest.mark.parametrize(
    "events,tab_widget_count",
    [
        pytest.param(
            [
                FullSnapshotEvent(
                    snapshot=(
                        SnapshotBuilder()
                        .add_step(step_id="0", status=state.STEP_STATE_UNKNOWN)
                        .add_job(
                            step_id="0",
                            job_id="0",
                            index="0",
                            name="job_0",
                            status=state.JOB_STATE_START,
                        )
                        .build(["0"], state.REALIZATION_STATE_UNKNOWN)
                    ),
                    phase_name="Foo",
                    current_phase=0,
                    total_phases=1,
                    progress=0.25,
                    iteration=0,
                    indeterminate=False,
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
                    iteration=0,
                    indeterminate=False,
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
                        .add_step(step_id="0", status=state.STEP_STATE_UNKNOWN)
                        .add_job(
                            step_id="0",
                            job_id="0",
                            index="0",
                            name="job_0",
                            max_memory_usage="1000",
                            current_memory_usage="500",
                            status=state.JOB_STATE_START,
                        )
                        .build(["0"], state.REALIZATION_STATE_UNKNOWN)
                    ),
                    phase_name="Foo",
                    current_phase=0,
                    total_phases=1,
                    progress=0.25,
                    iteration=0,
                    indeterminate=False,
                ),
                SnapshotUpdateEvent(
                    partial_snapshot=PartialSnapshot(
                        SnapshotBuilder()
                        .add_step(step_id="0", status=state.STEP_STATE_SUCCESS)
                        .build(["0"], status=state.REALIZATION_STATE_FINISHED)
                    ),
                    phase_name="Foo",
                    current_phase=0,
                    total_phases=1,
                    progress=0.5,
                    iteration=0,
                    indeterminate=False,
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
                        .add_step(step_id="0", status=state.STEP_STATE_UNKNOWN)
                        .add_job(
                            step_id="0",
                            job_id="0",
                            index="0",
                            name="job_0",
                            status=state.JOB_STATE_START,
                        )
                        .add_job(
                            step_id="0",
                            job_id="1",
                            index="1",
                            name="job_1",
                            status=state.JOB_STATE_START,
                        )
                        .build(["0", "1"], state.REALIZATION_STATE_UNKNOWN)
                    ),
                    phase_name="Foo",
                    current_phase=0,
                    total_phases=1,
                    progress=0.25,
                    iteration=0,
                    indeterminate=False,
                ),
                SnapshotUpdateEvent(
                    partial_snapshot=PartialSnapshot(
                        SnapshotBuilder()
                        .add_step(step_id="0", status=state.STEP_STATE_SUCCESS)
                        .add_job(
                            step_id="0",
                            job_id="0",
                            index="0",
                            status=state.JOB_STATE_FINISHED,
                            name="job_0",
                        )
                        .build(["1"], status=state.REALIZATION_STATE_RUNNING)
                    ),
                    phase_name="Foo",
                    current_phase=0,
                    total_phases=1,
                    progress=0.5,
                    iteration=0,
                    indeterminate=False,
                ),
                SnapshotUpdateEvent(
                    partial_snapshot=PartialSnapshot(
                        SnapshotBuilder()
                        .add_step(step_id="0", status=state.STEP_STATE_FAILURE)
                        .add_job(
                            step_id="0",
                            job_id="1",
                            index="1",
                            status=state.JOB_STATE_FAILURE,
                            name="job_1",
                        )
                        .build(["0"], status=state.REALIZATION_STATE_FAILED)
                    ),
                    phase_name="Foo",
                    current_phase=0,
                    total_phases=1,
                    progress=0.5,
                    iteration=0,
                    indeterminate=False,
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
                        .add_step(step_id="0", status=state.STEP_STATE_UNKNOWN)
                        .add_job(
                            step_id="0",
                            job_id="0",
                            index="0",
                            name="job_0",
                            status=state.JOB_STATE_START,
                        )
                        .build(["0"], state.REALIZATION_STATE_UNKNOWN)
                    ),
                    phase_name="Foo",
                    current_phase=0,
                    total_phases=1,
                    progress=0.25,
                    iteration=0,
                    indeterminate=False,
                ),
                FullSnapshotEvent(
                    snapshot=(
                        SnapshotBuilder()
                        .add_step(step_id="0", status=state.STEP_STATE_UNKNOWN)
                        .add_job(
                            step_id="0",
                            job_id="0",
                            index="0",
                            name="job_0",
                            status=state.JOB_STATE_START,
                        )
                        .build(["0"], state.REALIZATION_STATE_UNKNOWN)
                    ),
                    phase_name="Foo",
                    current_phase=0,
                    total_phases=1,
                    progress=0.5,
                    iteration=1,
                    indeterminate=False,
                ),
                EndEvent(failed=False, failed_msg=""),
            ],
            2,
            id="two_iterations",
        ),
    ],
)
def test_run_dialog(events, tab_widget_count, runmodel, qtbot: QtBot, mock_tracker):
    notifier = Mock()
    widget = RunDialog("poly.ert", runmodel, notifier)
    widget.show()
    qtbot.addWidget(widget)

    with patch("ert.gui.simulation.run_dialog.EvaluatorTracker") as tracker:
        tracker.return_value = mock_tracker(events)
        widget.startSimulation()

    with qtbot.waitExposed(widget, timeout=30000):
        qtbot.mouseClick(widget.show_details_button, Qt.LeftButton)
        qtbot.waitUntil(
            lambda: widget._tab_widget.count() == tab_widget_count, timeout=5000
        )
        qtbot.waitUntil(widget.done_button.isVisible, timeout=5000)


@pytest.mark.usefixtures("copy_poly_case")
def test_that_run_dialog_can_be_closed_while_file_plot_is_open(
    qtbot: QtBot, storage, source_root
):
    """
    This is a regression test for a crash happening when
    closing the RunDialog with a file open.
    """

    config_file = Path("poly.ert")
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

        def handle_dialog():
            message_box = gui.findChild(QMessageBox)
            qtbot.mouseClick(message_box.button(QMessageBox.Yes), Qt.LeftButton)

        QTimer.singleShot(500, handle_dialog)
        qtbot.mouseClick(start_simulation, Qt.LeftButton)

        qtbot.waitUntil(lambda: gui.findChild(RunDialog) is not None)
        run_dialog = gui.findChild(RunDialog)
        qtbot.mouseClick(run_dialog.show_details_button, Qt.LeftButton)
        job_view = run_dialog._job_view
        qtbot.waitUntil(job_view.isVisible, timeout=20000)
        qtbot.waitUntil(run_dialog.done_button.isVisible, timeout=20000)

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
        click_pos = job_view.visualRect(run_dialog._job_model.index(0, 4)).center()
        qtbot.mouseClick(job_view.viewport(), Qt.LeftButton, pos=click_pos)

        qtbot.waitUntil(run_dialog.findChild(FileDialog).isVisible, timeout=3000)

        with qtbot.waitSignal(run_dialog.accepted, timeout=30000):
            run_dialog.close()  # Close the run dialog by pressing 'x' close button

        # Ensure that once the run dialog is closed
        # another simulation can be started
        assert start_simulation.isEnabled()
