from datetime import datetime

import pytest
from cloudevents.http.event import CloudEvent

from ert.ensemble_evaluator import identifiers as ids
from ert.ensemble_evaluator import state
from ert.ensemble_evaluator.snapshot import (
    ForwardModel,
    PartialSnapshot,
    Snapshot,
    SnapshotBuilder,
    _get_job_id,
    _get_real_id,
)


def test_snapshot_merge(snapshot: Snapshot):
    update_event = PartialSnapshot(snapshot)
    update_event.update_job(
        real_id="1",
        job_id="0",
        job=ForwardModel(
            status="Finished",
            index="0",
            start_time=datetime(year=2020, month=10, day=27),
            end_time=datetime(year=2020, month=10, day=28),
        ),
    )
    update_event.update_job(
        real_id="1",
        job_id="1",
        job=ForwardModel(
            status="Running",
            index="1",
            start_time=datetime(year=2020, month=10, day=27),
        ),
    )
    update_event.update_job(
        real_id="9",
        job_id="0",
        job=ForwardModel(
            status="Running",
            index="0",
            start_time=datetime(year=2020, month=10, day=27),
        ),
    )

    snapshot.merge_event(update_event)

    assert snapshot.status == state.ENSEMBLE_STATE_UNKNOWN

    assert snapshot.get_job(real_id="1", job_id="0") == ForwardModel(
        status="Finished",
        index="0",
        start_time=datetime(year=2020, month=10, day=27),
        end_time=datetime(year=2020, month=10, day=28),
        name="job0",
    )

    assert snapshot.get_job(real_id="1", job_id="1") == ForwardModel(
        status="Running",
        index="1",
        start_time=datetime(year=2020, month=10, day=27),
        name="job1",
    )

    assert snapshot.get_job(real_id="9", job_id="0").status == "Running"
    assert snapshot.get_job(real_id="9", job_id="0") == ForwardModel(
        status="Running",
        index="0",
        start_time=datetime(year=2020, month=10, day=27),
        name="job0",
    )


@pytest.mark.parametrize(
    "source_string, expected_ids",
    [
        (
            "/ert/ee/0/real/1111/job/0",
            {"real": "1111", "job": "0"},
        ),
        (
            "/ert/ee/0/real/1111",
            {"real": "1111", "job": None},
        ),
        (
            "/ert/ee/0/real/1111",
            {"real": "1111", "job": None},
        ),
        (
            "/ert/ee/0/real/1111",
            {"real": "1111", "job": None},
        ),
        (
            "/ert/ee/0",
            {"real": None, "job": None},
        ),
    ],
)
def test_source_get_ids(source_string, expected_ids):
    assert _get_real_id(source_string) == expected_ids["real"]
    assert _get_job_id(source_string) == expected_ids["job"]


def test_update_jobs_in_partial_from_multiple_cloudevents(snapshot):
    partial = PartialSnapshot(snapshot)
    partial.from_cloudevent(
        CloudEvent(
            attributes={
                "id": "0",
                "type": ids.EVTYPE_FM_JOB_RUNNING,
                "source": "/real/0/job/0",
            },
            data={
                "current_memory_usage": 5,
                "max_memory_usage": 6,
            },
        )
    )
    partial.from_cloudevent(
        CloudEvent(
            {
                "id": "0",
                "type": ids.EVTYPE_FM_JOB_FAILURE,
                "source": "/real/0/job/0",
            },
            {ids.ERROR_MSG: "failed"},
        )
    )
    partial.from_cloudevent(
        CloudEvent(
            {
                "id": "1",
                "type": ids.EVTYPE_FM_JOB_SUCCESS,
                "source": "/real/0/job/1",
            }
        )
    )
    jobs = partial.to_dict()["reals"]["0"]["jobs"]
    assert jobs["0"]["status"] == state.JOB_STATE_FAILURE
    assert jobs["1"]["status"] == state.JOB_STATE_FINISHED


def test_that_realization_success_message_updates_state(snapshot):
    snapshot = SnapshotBuilder().build(["0"], status="Unknown")
    partial = PartialSnapshot(snapshot)
    partial.from_cloudevent(
        CloudEvent(
            {
                "id": "0",
                "type": ids.EVTYPE_REALIZATION_SUCCESS,
                "source": "/real/0",
            }
        )
    )
    assert partial.to_dict()["reals"]["0"]["status"] == state.REALIZATION_STATE_FINISHED
