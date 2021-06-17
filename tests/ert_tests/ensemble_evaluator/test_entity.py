import logging
from datetime import datetime

import pytest
from cloudevents.http.event import CloudEvent

import ert_shared.status.entity.state as state
from ert_shared.ensemble_evaluator.entity import command
from ert_shared.ensemble_evaluator.entity import identifiers as ids
from ert_shared.ensemble_evaluator.entity import tool
from ert_shared.ensemble_evaluator.ensemble.base import _EnsembleStateTracker
from ert_shared.ensemble_evaluator.entity.snapshot import (
    Job,
    PartialSnapshot,
    SnapshotBuilder,
)


def test_snapshot_merge(snapshot):
    update_event = PartialSnapshot(snapshot)
    update_event.update_status(status=state.ENSEMBLE_STATE_STARTED)

    snapshot.merge_event(update_event)

    assert snapshot.get_status() == state.ENSEMBLE_STATE_STARTED

    update_event = PartialSnapshot(snapshot)
    update_event.update_job(
        real_id="1",
        step_id="0",
        job_id="0",
        job=Job(
            status="Finished",
            start_time=datetime(year=2020, month=10, day=27),
            end_time=datetime(year=2020, month=10, day=28),
            data={"memory": 1000},
        ),
    )
    update_event.update_job(
        real_id="1",
        step_id="0",
        job_id="1",
        job=Job(
            status="Running",
            start_time=datetime(year=2020, month=10, day=27),
        ),
    )
    update_event.update_job(
        real_id="9",
        step_id="0",
        job_id="0",
        job=Job(
            status="Running",
            start_time=datetime(year=2020, month=10, day=27),
        ),
    )

    snapshot.merge_event(update_event)

    assert snapshot.get_status() == state.ENSEMBLE_STATE_STARTED

    assert snapshot.get_job(real_id="1", step_id="0", job_id="0") == Job(
        status="Finished",
        start_time=datetime(year=2020, month=10, day=27),
        end_time=datetime(year=2020, month=10, day=28),
        data={"memory": 1000},
        error=None,
        name="job0",
        stderr=None,
        stdout=None,
    )

    assert snapshot.get_job(real_id="1", step_id="0", job_id="1") == Job(
        status="Running",
        start_time=datetime(year=2020, month=10, day=27),
        end_time=None,
        data={},
        error=None,
        name="job1",
        stderr=None,
        stdout=None,
    )

    assert snapshot.get_job(real_id="9", step_id="0", job_id="0").status == "Running"
    assert snapshot.get_job(real_id="9", step_id="0", job_id="0") == Job(
        status="Running",
        start_time=datetime(year=2020, month=10, day=27),
        end_time=None,
        data={},
        error=None,
        name="job0",
        stderr=None,
        stdout=None,
    )


@pytest.mark.parametrize(
    "source_string, expected_ids",
    [
        (
            "/ert/ee/1234/real/1111/step/asd123ASD/job/0",
            {"evaluation_id": "1234", "real": "1111", "step": "asd123ASD", "job": "0"},
        ),
        (
            "/ert/ee/some_eval_id/real/1111/step/asd123ASD",
            {
                "evaluation_id": "some_eval_id",
                "real": "1111",
                "step": "asd123ASD",
                "job": None,
            },
        ),
        (
            "/ert/ee/0/real/1111",
            {"evaluation_id": "0", "real": "1111", "step": None, "job": None},
        ),
        (
            "/ert/ee/a1ff5a4d8/real/1111",
            {"evaluation_id": "a1ff5a4d8", "real": "1111", "step": None, "job": None},
        ),
        (
            "/ert/ee/ee_id_0",
            {"evaluation_id": "ee_id_0", "real": None, "step": None, "job": None},
        ),
    ],
)
def test_source_get_ids(source_string, expected_ids):

    assert tool.get_evaluation_id(source_string) == expected_ids["evaluation_id"]
    assert tool.get_real_id(source_string) == expected_ids["real"]
    assert tool.get_step_id(source_string) == expected_ids["step"]
    assert tool.get_job_id(source_string) == expected_ids["job"]


def test_commands_to_and_from_dict():
    pause_command = command.create_command_pause()
    terminate_command = command.create_command_terminate()

    assert pause_command == command.create_command_from_dict(pause_command.to_dict())
    assert terminate_command == command.create_command_from_dict(
        terminate_command.to_dict()
    )


def test_update_partial_from_multiple_cloudevents(snapshot):
    partial = PartialSnapshot(snapshot)
    partial.from_cloudevent(
        CloudEvent(
            {
                "id": "0",
                "type": ids.EVTYPE_FM_JOB_RUNNING,
                "source": "/real/0/step/0/job/0",
            }
        )
    )
    partial.from_cloudevent(
        CloudEvent(
            {
                "id": "0",
                "type": ids.EVTYPE_FM_JOB_FAILURE,
                "source": "/real/0/step/0/job/0",
            },
            {ids.ERROR_MSG: "failed"},
        )
    )
    partial.from_cloudevent(
        CloudEvent(
            {
                "id": "1",
                "type": ids.EVTYPE_FM_JOB_SUCCESS,
                "source": "/real/0/step/0/job/1",
            }
        )
    )
    jobs = partial.to_dict()["reals"]["0"]["steps"]["0"]["jobs"]
    jobs["0"]["status"] == state.JOB_STATE_FAILURE
    jobs["1"]["status"] == state.JOB_STATE_FINISHED


def test_multiple_cloud_events_trigger_non_communicated_change():
    """In other words, though we say all steps are finished, we don't
    explicitly send an event that changes the realization status. It should
    happen by virtue of the steps being completed."""
    snapshot = (
        SnapshotBuilder()
        .add_step(step_id="0", status="Unknown")
        .build(["0"], status="Unknown")
    )
    partial = PartialSnapshot(snapshot)
    partial.from_cloudevent(
        CloudEvent(
            {
                "id": "0",
                "type": ids.EVTYPE_FM_STEP_SUCCESS,
                "source": "/real/0/step/0",
            }
        )
    )
    assert partial.to_dict()["reals"]["0"]["status"] == state.REALIZATION_STATE_FINISHED


@pytest.mark.parametrize(
    "transition, allowed",
    [
        ([state.ENSEMBLE_STATE_STARTED, state.ENSEMBLE_STATE_STOPPED], True),
        ([state.ENSEMBLE_STATE_STARTED, state.ENSEMBLE_STATE_FAILED], True),
        ([state.ENSEMBLE_STATE_STARTED, state.ENSEMBLE_STATE_CANCELLED], True),
        ([state.ENSEMBLE_STATE_CANCELLED, state.ENSEMBLE_STATE_STARTED], False),
        ([state.ENSEMBLE_STATE_CANCELLED, state.ENSEMBLE_STATE_STOPPED], False),
        ([state.ENSEMBLE_STATE_CANCELLED, state.ENSEMBLE_STATE_FAILED], False),
        ([state.ENSEMBLE_STATE_STOPPED, state.ENSEMBLE_STATE_FAILED], False),
        ([state.ENSEMBLE_STATE_STOPPED, state.ENSEMBLE_STATE_CANCELLED], False),
        ([state.ENSEMBLE_STATE_STOPPED, state.ENSEMBLE_STATE_STARTED], False),
        ([state.ENSEMBLE_STATE_FAILED, state.ENSEMBLE_STATE_STARTED], False),
        ([state.ENSEMBLE_STATE_FAILED, state.ENSEMBLE_STATE_STOPPED], False),
        ([state.ENSEMBLE_STATE_FAILED, state.ENSEMBLE_STATE_CANCELLED], False),
        ([state.ENSEMBLE_STATE_UNKNOWN, state.ENSEMBLE_STATE_STARTED], True),
    ],
)
def test_ensemble_state_tracker(transition, allowed, caplog, snapshot):
    initial_state, update_state = transition
    with caplog.at_level(logging.WARNING):
        state_tracker = _EnsembleStateTracker(initial_state)
        new_state = state_tracker.update_state(update_state)
        assert new_state == update_state
        if allowed:
            assert len(caplog.records) == 0
        else:
            assert len(caplog.records) == 1
            log_mgs = f"Illegal state transition from {initial_state} to {update_state}"
            assert log_mgs == caplog.records[0].msg


def test_ensemble_state_tracker_handles():
    state_machine = _EnsembleStateTracker()
    expected_sates = [
        state.ENSEMBLE_STATE_UNKNOWN,
        state.ENSEMBLE_STATE_STARTED,
        state.ENSEMBLE_STATE_FAILED,
        state.ENSEMBLE_STATE_STOPPED,
        state.ENSEMBLE_STATE_CANCELLED,
    ]
    handled_states = list(state_machine._handles.keys())
    assert len(handled_states) == len(expected_sates)
    for handled_state in handled_states:
        assert handled_state in expected_sates
