import time
from unittest.mock import Mock, patch

import pytest
from ert_gui.simulation.run_dialog import RunDialog
from ert_shared.ensemble_evaluator.entity.snapshot import (
    Snapshot,
    SnapshotBuilder,
)
from ert_shared.status.entity.event import (
    EndEvent,
    FullSnapshotEvent,
)
from ert_shared.status.entity.state import (
    JOB_STATE_START,
    REALIZATION_STATE_UNKNOWN,
    STAGE_STATE_UNKNOWN,
    STEP_STATE_START,
)
from qtpy.QtCore import Qt


@pytest.fixture
def runmodel() -> Mock:
    brm = Mock()
    brm.get_runtime = Mock(return_value=100)
    brm.hasRunFailed = Mock(return_value=False)
    brm.getFailMessage = Mock(return_value="")
    brm.support_restart = True
    return brm


@pytest.fixture
def active_realizations() -> Mock:
    active_reals = Mock()
    active_reals.count = Mock(return_value=10)
    return active_reals


@pytest.fixture()
def full_snapshot() -> Snapshot:
    builder = (
        SnapshotBuilder()
        .add_stage(stage_id="0", status=STAGE_STATE_UNKNOWN)
        .add_step(stage_id="0", step_id="0", status=STEP_STATE_START)
    )
    for i in range(0, 150):
        builder.add_job(
            stage_id="0",
            step_id="0",
            job_id=str(i),
            name=f"job_{i}",
            data={},
            status=JOB_STATE_START,
        )
    real_ids = [str(i) for i in range(0, 150)]
    return builder.build(real_ids, REALIZATION_STATE_UNKNOWN)


class MockTracker:
    def __init__(self, events) -> None:
        self._events = events

    def track(self):
        for event in self._events:
            yield event
            time.sleep(0.1)

    def reset(self):
        pass


def test_success(runmodel, active_realizations, qtbot):
    widget = RunDialog(
        "poly.ert", runmodel, {"active_realizations": active_realizations}
    )
    widget.has_failed_realizations = lambda: False
    widget.show()
    qtbot.addWidget(widget)

    with patch("ert_gui.simulation.run_dialog.create_tracker") as mock_tracker_factory:
        mock_tracker_factory.return_value = MockTracker(
            [EndEvent(failed=False, failed_msg="")]
        )
        widget.startSimulation()

    qtbot.waitForWindowShown(widget)
    assert widget._total_progress_bar.value() == 100
    assert widget.done_button.isVisible()
    assert widget.done_button.text() == "Done"


def test_full_snapshot(runmodel, active_realizations, full_snapshot, qtbot):
    widget = RunDialog(
        "poly.ert", runmodel, {"active_realizations": active_realizations}
    )
    widget.has_failed_realizations = lambda: False
    widget.show()
    qtbot.addWidget(widget)

    with patch("ert_gui.simulation.run_dialog.create_tracker") as mock_tracker_factory:
        iter_0 = FullSnapshotEvent(
            snapshot=full_snapshot,
            phase_name="Foo",
            current_phase=0,
            total_phases=1,
            progress=0.5,
            iteration=0,
            indeterminate=False,
        )
        iter_1 = FullSnapshotEvent(
            snapshot=full_snapshot,
            phase_name="Foo",
            current_phase=0,
            total_phases=1,
            progress=0.5,
            iteration=1,
            indeterminate=False,
        )
        mock_tracker_factory.return_value = MockTracker(
            [iter_0, iter_1, EndEvent(failed=False, failed_msg="")]
        )
        widget.startSimulation()

    qtbot.waitForWindowShown(widget)
    assert widget._total_progress_bar.value() == 50
    qtbot.mouseClick(widget.show_details_button, Qt.LeftButton)
    qtbot.waitSignal(widget._snapshot_model.rowsInserted)
    qtbot.waitSignal(widget._snapshot_model.rowsInserted)
    assert widget._tab_widget.count() == 2
