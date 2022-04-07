from unittest.mock import patch

import ert_shared.ensemble_evaluator.entity.identifiers as ids
import pytest
from ert_gui.simulation.run_dialog import RunDialog
from ert_shared.ensemble_evaluator.entity.snapshot import (
    PartialSnapshot,
    SnapshotBuilder,
)
from ert_shared.status.entity import state
from ert_shared.status.entity.event import (
    EndEvent,
    FullSnapshotEvent,
    SnapshotUpdateEvent,
)
from qtpy.QtCore import Qt


def test_success(runmodel, qtbot, mock_tracker):
    widget = RunDialog("poly.ert", runmodel)
    widget.show()
    qtbot.addWidget(widget)

    with patch("ert_gui.simulation.run_dialog.create_tracker") as mock_tracker_factory:
        mock_tracker_factory.return_value = mock_tracker(
            [EndEvent(failed=False, failed_msg="")]
        )
        widget.startSimulation()

    qtbot.waitForWindowShown(widget)
    qtbot.waitUntil(lambda: widget._total_progress_bar.value() == 100)
    assert widget.done_button.isVisible()
    assert widget.done_button.text() == "Done"


def test_large_snapshot(runmodel, large_snapshot, qtbot, mock_tracker):
    widget = RunDialog("poly.ert", runmodel)
    widget.show()
    qtbot.addWidget(widget)

    with patch("ert_gui.simulation.run_dialog.create_tracker") as mock_tracker_factory:
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
        mock_tracker_factory.return_value = mock_tracker(
            [iter_0, iter_1, EndEvent(failed=False, failed_msg="")]
        )
        widget.startSimulation()

    qtbot.waitForWindowShown(widget)
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

    with patch("ert_gui.simulation.run_dialog.create_tracker") as mock_tracker_factory:
        mock_tracker_factory.return_value = mock_tracker(events)
        widget.startSimulation()

    qtbot.waitForWindowShown(widget)
    qtbot.mouseClick(widget.show_details_button, Qt.LeftButton)
    qtbot.waitUntil(
        lambda: widget._tab_widget.count() == tab_widget_count, timeout=5000
    )
    qtbot.waitUntil(lambda: widget.done_button.isVisible(), timeout=5000)
