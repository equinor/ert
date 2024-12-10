import logging
from datetime import datetime

from _ert.events import (
    ForwardModelStepFailure,
    ForwardModelStepRunning,
    ForwardModelStepStart,
    ForwardModelStepSuccess,
    RealizationSuccess,
)
from ert.ensemble_evaluator import state
from ert.ensemble_evaluator.snapshot import (
    EnsembleSnapshot,
    FMStepSnapshot,
)
from tests.ert import SnapshotBuilder


def test_snapshot_merge(snapshot: EnsembleSnapshot):
    update_event = EnsembleSnapshot()
    update_event.update_fm_step(
        real_id="1",
        fm_step_id="0",
        fm_step=FMStepSnapshot(
            status="Finished",
            index="0",
            start_time=datetime(year=2020, month=10, day=27),
            end_time=datetime(year=2020, month=10, day=28),
        ),
    )
    update_event.update_fm_step(
        real_id="1",
        fm_step_id="1",
        fm_step=FMStepSnapshot(
            status="Running",
            index="1",
            start_time=datetime(year=2020, month=10, day=27),
        ),
    )
    update_event.update_fm_step(
        real_id="9",
        fm_step_id="0",
        fm_step=FMStepSnapshot(
            status="Running",
            index="0",
            start_time=datetime(year=2020, month=10, day=27),
        ),
    )

    snapshot.merge_snapshot(update_event)

    assert snapshot.status == state.ENSEMBLE_STATE_UNKNOWN

    assert snapshot.get_fm_step(real_id="1", fm_step_id="0") == FMStepSnapshot(
        status="Finished",
        index="0",
        start_time=datetime(year=2020, month=10, day=27),
        end_time=datetime(year=2020, month=10, day=28),
        name="forward_model0",
    )

    assert snapshot.get_fm_step(real_id="1", fm_step_id="1") == FMStepSnapshot(
        status="Running",
        index="1",
        start_time=datetime(year=2020, month=10, day=27),
        name="forward_model1",
    )

    assert snapshot.get_fm_step(real_id="9", fm_step_id="0").get("status") == "Running"
    assert snapshot.get_fm_step(real_id="9", fm_step_id="0") == FMStepSnapshot(
        status="Running",
        index="0",
        start_time=datetime(year=2020, month=10, day=27),
        name="forward_model0",
    )


def test_update_forward_models_in_partial_from_multiple_messages(snapshot):
    new_snapshot = EnsembleSnapshot()
    new_snapshot.update_from_event(
        ForwardModelStepRunning(
            ensemble="1",
            real="0",
            fm_step="0",
            current_memory_usage=5,
            max_memory_usage=6,
        )
    )
    new_snapshot.update_from_event(
        ForwardModelStepFailure(
            ensemble="1", real="0", fm_step="0", error_msg="failed"
        ),
    )
    new_snapshot.update_from_event(
        ForwardModelStepSuccess(ensemble="1", real="0", fm_step="1")
    )
    forward_models = new_snapshot.to_dict()["reals"]["0"]["fm_steps"]
    assert forward_models["0"]["status"] == state.FORWARD_MODEL_STATE_FAILURE
    assert forward_models["1"]["status"] == state.FORWARD_MODEL_STATE_FINISHED


def test_that_realization_success_message_updates_state():
    snapshot = SnapshotBuilder().build(["0"], status="Unknown")
    snapshot.update_from_event(RealizationSuccess(ensemble="0", real="0"))
    assert (
        snapshot.to_dict()["reals"]["0"]["status"] == state.REALIZATION_STATE_FINISHED
    )


def test_fm_updates_previous_fm_if_it_is_stuck_in_nonfinalized_state(caplog):
    """In case a snapshot event is lost (if connection is dropped), a forwardmodel can be stuck in a non-finalized state.
    This should be handled, and it is safe to assume it has already finished if the next forwardmodel has started.
    We also don't know the end_time of the prior forwardmodel if we have lost a finished/exited event,
    so we set it to the start_time of the next forwardmodel."""
    caplog.set_level(logging.ERROR)

    main_snapshot = EnsembleSnapshot()
    main_snapshot._fm_step_snapshots["0", "0"] = FMStepSnapshot(
        status="Pending",
        index="0",
        start_time=datetime.fromtimestamp(757575),
        name="forward_model0",
    )
    main_snapshot._fm_step_snapshots["0", "1"] = FMStepSnapshot(
        status="Running",
        index="1",
        start_time=datetime.fromtimestamp(939393),
        name="forward_model1",
    )

    update_snapshot = EnsembleSnapshot()
    update_snapshot.update_from_event(
        ForwardModelStepStart(
            real="0", fm_step="2", time=datetime.fromtimestamp(10101010)
        ),
        main_snapshot,
    )
    update_snapshot.update_from_event(
        ForwardModelStepRunning(real="0", fm_step="2"), main_snapshot
    )
    main_snapshot.merge_snapshot(update_snapshot)

    affected_snapshots = [main_snapshot, update_snapshot]
    for snapshot in affected_snapshots:
        assert snapshot.get_fm_step(real_id="0", fm_step_id="0")["status"] == "Finished"
        assert snapshot.get_fm_step(real_id="0", fm_step_id="0")[
            "end_time"
        ] == datetime.fromtimestamp(939393)

        assert snapshot.get_fm_step(real_id="0", fm_step_id="1")["status"] == "Finished"
        assert snapshot.get_fm_step(real_id="0", fm_step_id="1")[
            "end_time"
        ] == datetime.fromtimestamp(10101010)

        assert snapshot.get_fm_step(real_id="0", fm_step_id="2")["status"] == "Running"
        assert "end_time" not in snapshot.get_fm_step(real_id="0", fm_step_id="2")

    assert "Did not get finished event for" in caplog.text
