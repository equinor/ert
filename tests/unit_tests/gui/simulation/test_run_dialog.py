from unittest.mock import patch

import pytest
from qtpy import QtWidgets
from qtpy.QtCore import Qt, QTimer

from ert.ensemble_evaluator import identifiers as ids
from ert.ensemble_evaluator import state
from ert.ensemble_evaluator.event import (
    EndEvent,
    FullSnapshotEvent,
    SnapshotUpdateEvent,
)
from ert.ensemble_evaluator.snapshot import PartialSnapshot, SnapshotBuilder
from ert.gui.simulation.run_dialog import RunDialog


def test_success(runmodel, qtbot, mock_tracker):
    widget = RunDialog("poly.ert", runmodel)
    widget.show()
    qtbot.addWidget(widget)

    with patch("ert.gui.simulation.run_dialog.EvaluatorTracker") as tracker:
        tracker.return_value = mock_tracker([EndEvent(failed=False, failed_msg="")])
        widget.startSimulation()

    with qtbot.waitExposed(widget, timeout=30000):
        qtbot.waitUntil(lambda: widget._total_progress_bar.value() == 100)
        assert widget.done_button.isVisible()
        assert widget.done_button.text() == "Done"


# pylint: disable=no-member
def test_kill_simulations(runmodel, qtbot, mock_tracker):
    widget = RunDialog("poly.ert", runmodel)
    widget.show()
    qtbot.addWidget(widget)

    with patch("ert.gui.simulation.run_dialog.EvaluatorTracker") as tracker:
        tracker.return_value = mock_tracker([EndEvent(failed=False, failed_msg="")])
        widget.startSimulation()

    with qtbot.waitExposed(widget, timeout=30000):

        def handle_dialog():
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


def test_large_snapshot(runmodel, large_snapshot, qtbot, mock_tracker):
    widget = RunDialog("poly.ert", runmodel)
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

    with qtbot.waitExposed(widget, timeout=30000):
        qtbot.waitUntil(lambda: widget._total_progress_bar.value() == 100, timeout=5000)
        qtbot.mouseClick(widget.show_details_button, Qt.LeftButton)
        qtbot.waitUntil(lambda: widget._tab_widget.count() == 2, timeout=5000)


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
                            data={},
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
                            data={
                                ids.MAX_MEMORY_USAGE: 1000,
                                ids.CURRENT_MEMORY_USAGE: 500,
                            },
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
                            data={},
                            status=state.JOB_STATE_START,
                        )
                        .add_job(
                            step_id="0",
                            job_id="1",
                            index="1",
                            name="job_1",
                            data={},
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
                            data={},
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
                            data={},
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
                            data={},
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
                            data={},
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
def test_run_dialog(events, tab_widget_count, runmodel, qtbot, mock_tracker):
    widget = RunDialog("poly.ert", runmodel)
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
