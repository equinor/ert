from datetime import datetime

import pytest
from cloudevents.http.event import CloudEvent

from ert.ensemble_evaluator import identifiers as ids
from ert.ensemble_evaluator import state
from ert.ensemble_evaluator.snapshot import (
    Job,
    PartialSnapshot,
    Snapshot,
    SnapshotBuilder,
    _get_job_id,
    _get_real_id,
    _get_step_id,
)


def test_snapshot_merge(snapshot: Snapshot):
    update_event = PartialSnapshot(snapshot)
    update_event.update_status(status=state.ENSEMBLE_STATE_STARTED)

    snapshot.merge_event(update_event)

    assert snapshot.status == state.ENSEMBLE_STATE_STARTED

    update_event = PartialSnapshot(snapshot)
    update_event.update_job(
        real_id="1",
        step_id="0",
        job_id="0",
        job=Job(
            status="Finished",
            index="0",
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
            index="1",
            start_time=datetime(year=2020, month=10, day=27),
        ),
    )
    update_event.update_job(
        real_id="9",
        step_id="0",
        job_id="0",
        job=Job(
            status="Running",
            index="0",
            start_time=datetime(year=2020, month=10, day=27),
        ),
    )

    snapshot.merge_event(update_event)

    assert snapshot.status == state.ENSEMBLE_STATE_STARTED

    assert snapshot.get_job(real_id="1", step_id="0", job_id="0") == Job(
        status="Finished",
        index="0",
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
        index="1",
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
        index="0",
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
            "/ert/ee/0/real/1111/step/asd123ASD/job/0",
            {"real": "1111", "step": "asd123ASD", "job": "0"},
        ),
        (
            "/ert/ee/0/real/1111/step/asd123ASD",
            {"real": "1111", "step": "asd123ASD", "job": None},
        ),
        (
            "/ert/ee/0/real/1111",
            {"real": "1111", "step": None, "job": None},
        ),
        (
            "/ert/ee/0/real/1111",
            {"real": "1111", "step": None, "job": None},
        ),
        (
            "/ert/ee/0",
            {"real": None, "step": None, "job": None},
        ),
    ],
)
def test_source_get_ids(source_string, expected_ids):
    assert _get_real_id(source_string) == expected_ids["real"]
    assert _get_step_id(source_string) == expected_ids["step"]
    assert _get_job_id(source_string) == expected_ids["job"]


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
    assert jobs["0"]["status"] == state.JOB_STATE_FAILURE
    assert jobs["1"]["status"] == state.JOB_STATE_FINISHED


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
