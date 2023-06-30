from datetime import datetime, timedelta

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

STEP = "0"


def fm_job_start_event(
    realization: int, forward_model: int, start_time: datetime
) -> CloudEvent:
    return CloudEvent(
        attributes={
            "specversion": "1.0",
            "id": "eventid",
            "source": (
                f"/ert/ensemble/ensemble-id/real/{realization}"
                f"/step/{STEP}/job/{forward_model}/index/{forward_model}"
            ),
            "type": ids.EVTYPE_FM_JOB_START,
            "datacontenttype": "application/json",
            "time": start_time.isoformat(),
        },
        data={
            "stdout": (
                f"scratch/realization-{realization}/iter-{forward_model}"
                f"/job_name.stdout.{forward_model}"
            ),
            "stderr": (
                f"scratch/realization-{realization}/iter-{forward_model}"
                f"/job_name.stderr.{forward_model}"
            ),
        },
    )


def fm_job_running_event(
    realization: int,
    forward_model: int,
    timestamp: datetime,
    current_memory_usage: int,
    max_memory_usage: int,
) -> CloudEvent:
    return CloudEvent(
        attributes={
            "specversion": "1.0",
            "id": "eventid",
            "source": (
                f"/ert/ensemble/ensemble-id/real/{realization}"
                f"/step/{STEP}/job/{forward_model}/index/{forward_model}"
            ),
            "type": ids.EVTYPE_FM_JOB_RUNNING,
            "datacontenttype": "application/json",
            "time": timestamp.isoformat(),
        },
        data={
            "max_memory_usage": max_memory_usage,
            "current_memory_usage": current_memory_usage,
        },
    )


def fm_job_done_event(
    realization: int,
    forward_model: int,
    timestamp: datetime,
) -> CloudEvent:
    return CloudEvent(
        attributes={
            "specversion": "1.0",
            "id": "eventid",
            "source": (
                f"/ert/ensemble/ensemble-id/real/{realization}"
                f"/step/{STEP}/job/{forward_model}/index/{forward_model}"
            ),
            "type": ids.EVTYPE_FM_JOB_SUCCESS,
            "datacontenttype": "application/json",
            "time": timestamp.isoformat(),
        },
        data=None,
    )


def fm_job_failed_event(
    realization: int, forward_model: int, timestamp: datetime, err_msg: str
) -> Job:
    return CloudEvent(
        attributes={
            "type": ids.EVTYPE_FM_JOB_FAILURE,
            "specversion": "1.0",
            "id": "eventid",
            "source": (
                f"/ert/ensemble/ensemble-id/real/{realization}"
                f"/step/{STEP}/job/{forward_model}/index/{forward_model}"
            ),
            "datacontenttype": "application/json",
            "time": timestamp.isoformat(),
        },
        data={
            "error_msg": err_msg,
        },
    )


def apply_and_verify_start_event(
    snapshot: PartialSnapshot,
    realization: int,
    forward_model: int,
    start_time: datetime,
) -> Job:
    start_event_fm = fm_job_start_event(realization, forward_model, start_time)
    fm = f"{forward_model}"
    real = f"{realization}"

    if "reals" in snapshot.data() and real in snapshot.data().reals:
        assert fm not in snapshot.data().reals[real].steps[STEP].jobs

    snapshot.from_cloudevent(start_event_fm)

    assert real in snapshot.data().reals
    assert STEP in snapshot.data().reals[real].steps
    assert fm in snapshot.data().reals[real].steps[STEP].jobs
    fm_start = snapshot.data().reals[real].steps[STEP].jobs[fm]
    assert fm_start.index == fm
    assert fm_start.stdout == start_event_fm.data["stdout"]
    assert fm_start.stderr == start_event_fm.data["stderr"]
    assert fm_start.status == "Pending"
    assert fm_start.start_time == start_time

    return fm_start


def apply_and_verify_running_event(
    snapshot: PartialSnapshot,
    realization: int,
    forward_model: int,
    timestamp: datetime,
    current_mem_usage: int,
    max_mem_usage: int,
) -> Job:
    running_event = fm_job_running_event(
        realization, forward_model, timestamp, current_mem_usage, max_mem_usage
    )
    fm = f"{forward_model}"
    real = f"{realization}"

    # job should be there from before
    assert fm in snapshot.data().reals[real].steps[STEP].jobs
    previous_fm = snapshot.data().reals[real].steps[STEP].jobs[fm]

    snapshot.from_cloudevent(running_event)

    assert fm in snapshot.data().reals[real].steps[STEP].jobs
    updated_fm = snapshot.data().reals[real].steps[STEP].jobs[fm]

    # updated stuff
    assert updated_fm.status == "Running"
    assert updated_fm.data.current_memory_usage == current_mem_usage
    if (
        "data" in previous_fm
        and "max_memory_usage" in previous_fm.data
        and previous_fm.data.max_memory_usage < max_mem_usage
    ) or ("data" not in previous_fm or "max_memory_usage" not in previous_fm.data):
        assert updated_fm.data.max_memory_usage == max_mem_usage
    else:
        assert updated_fm.data.max_memory_usage == max_mem_usage

    # unchanged stuff
    assert updated_fm.index == previous_fm.index
    assert updated_fm.stdout == previous_fm.stdout
    assert updated_fm.stderr == previous_fm.stderr
    assert updated_fm.start_time == previous_fm.start_time

    return updated_fm


def apply_and_verify_done_event(
    snapshot: PartialSnapshot,
    realization: int,
    forward_model: int,
    timestamp: datetime,
) -> Job:
    done_event = fm_job_done_event(realization, forward_model, timestamp)
    fm = f"{forward_model}"
    real = f"{realization}"

    # job should be there from before
    assert fm in snapshot.data().reals[real].steps[STEP].jobs
    previous_fm = snapshot.data().reals[real].steps[STEP].jobs[fm]

    snapshot.from_cloudevent(done_event)

    assert fm in snapshot.data().reals[real].steps[STEP].jobs
    updated_fm = snapshot.data().reals[real].steps[STEP].jobs[fm]

    # updated stuff
    assert updated_fm.status == "Finished"
    assert updated_fm.end_time == timestamp

    # unchanged stuff
    assert updated_fm.data.current_memory_usage == previous_fm.data.current_memory_usage
    assert updated_fm.data.max_memory_usage == previous_fm.data.max_memory_usage
    assert updated_fm.index == previous_fm.index
    assert updated_fm.stdout == previous_fm.stdout
    assert updated_fm.stderr == previous_fm.stderr
    assert updated_fm.start_time == previous_fm.start_time

    return updated_fm


def apply_and_verify_failed_event(
    snapshot: PartialSnapshot,
    realization: int,
    forward_model: int,
    timestamp: datetime,
    err_msg: str,
) -> Job:
    failed_event = fm_job_failed_event(realization, forward_model, timestamp, err_msg)
    fm = f"{forward_model}"
    real = f"{realization}"

    # job should be there from before
    assert fm in snapshot.data().reals[real].steps[STEP].jobs
    previous_fm = snapshot.data().reals[real].steps[STEP].jobs[fm]

    snapshot.from_cloudevent(failed_event)

    assert fm in snapshot.data().reals[real].steps[STEP].jobs
    updated_fm = snapshot.data().reals[real].steps[STEP].jobs[fm]

    # updated stuff
    assert updated_fm.status == "Failed"
    assert updated_fm.end_time == timestamp
    assert updated_fm.error == err_msg

    # unchanged stuff
    assert updated_fm.data.current_memory_usage == previous_fm.data.current_memory_usage
    assert updated_fm.data.max_memory_usage == previous_fm.data.max_memory_usage
    assert updated_fm.index == previous_fm.index
    assert updated_fm.stdout == previous_fm.stdout
    assert updated_fm.stderr == previous_fm.stderr
    assert updated_fm.start_time == previous_fm.start_time

    return updated_fm


def verify_fm_unchanged(
    snapshot: PartialSnapshot, realization: int, expected_forward_model: Job
):
    real = f"{realization}"
    fm = expected_forward_model.index

    assert fm in snapshot.data().reals[real].steps[STEP].jobs
    actual_fm = snapshot.data().reals[real].steps[STEP].jobs[fm]

    assert expected_forward_model == actual_fm


def test_snapshot_from_cloudevent_for_fm_events(snapshot: Snapshot):
    mutating_snapshot = PartialSnapshot(snapshot)

    # START FM 0

    start_time_stamp = datetime.fromisoformat("1995-06-13")
    realization_0 = 0
    realization_1 = 1
    forward_model_0 = 0
    forward_model_1 = 1

    start_fm0 = apply_and_verify_start_event(
        mutating_snapshot, realization_0, forward_model_0, start_time_stamp
    )

    # START FM 0 FOR REAL 1
    apply_and_verify_start_event(
        mutating_snapshot, realization_1, forward_model_0, start_time_stamp
    )

    # START FM 1

    start_fm1 = apply_and_verify_start_event(
        mutating_snapshot, realization_0, forward_model_1, start_time_stamp
    )

    verify_fm_unchanged(mutating_snapshot, realization_0, start_fm0)

    # FM 0 RUNNING

    fm0_running = apply_and_verify_running_event(
        mutating_snapshot,
        realization_0,
        forward_model_0,
        start_time_stamp + timedelta(seconds=1),
        5000,
        5000,
    )

    verify_fm_unchanged(mutating_snapshot, realization_0, start_fm1)

    # FM 1 RUNNING
    fm1_running = apply_and_verify_running_event(
        mutating_snapshot,
        realization_0,
        forward_model_1,
        start_time_stamp + timedelta(seconds=2),
        2000,
        3000,
    )

    verify_fm_unchanged(mutating_snapshot, realization_0, fm0_running)

    # FM 0 RUNNING UPDATE

    apply_and_verify_running_event(
        mutating_snapshot,
        realization_0,
        forward_model_0,
        start_time_stamp + timedelta(seconds=1),
        8000,
        8000,
    )
    verify_fm_unchanged(mutating_snapshot, realization_0, fm1_running)

    # FM 0 DONE

    fm0_done = apply_and_verify_done_event(
        mutating_snapshot,
        realization_0,
        forward_model_0,
        start_time_stamp + timedelta(seconds=3),
    )
    verify_fm_unchanged(mutating_snapshot, realization_0, fm1_running)

    # FM 1 FAILED

    apply_and_verify_failed_event(
        mutating_snapshot,
        realization_0,
        forward_model_1,
        start_time_stamp + timedelta(seconds=5),
        "we failed",
    )
    verify_fm_unchanged(mutating_snapshot, realization_0, fm0_done)


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
