from datetime import datetime
import dateutil.parser

import pytest
from ert_shared.ensemble_evaluator.entity import command, tool
from ert_shared.ensemble_evaluator.entity.snapshot import (
    PartialSnapshot,
    SnapshotBuilder,
    Job,
)


_REALIZATION_INDEXES = ["0", "1", "3", "4", "5", "9"]


def _create_snapshot():
    return (
        SnapshotBuilder()
        .add_stage(stage_id="0", status="Unknown")
        .add_step(stage_id="0", step_id="0", status="Unknown")
        .add_job(
            stage_id="0",
            step_id="0",
            job_id="0",
            name="job0",
            data={},
            status="Unknown",
        )
        .add_job(
            stage_id="0",
            step_id="0",
            job_id="1",
            name="job1",
            data={},
            status="Unknown",
        )
        .add_job(
            stage_id="0",
            step_id="0",
            job_id="2",
            name="job2",
            data={},
            status="Unknown",
        )
        .add_job(
            stage_id="0",
            step_id="0",
            job_id="3",
            name="job3",
            data={},
            status="Unknown",
        )
        .build(_REALIZATION_INDEXES, status="Unknown")
    )


def test_snapshot_merge():
    snapshot = _create_snapshot()

    update_event = PartialSnapshot(snapshot)
    update_event.update_status(status="running")

    snapshot.merge_event(update_event)

    assert snapshot.get_status() == "running"

    update_event = PartialSnapshot(snapshot)
    update_event.update_job(
        real_id="1",
        stage_id="0",
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
        stage_id="0",
        step_id="0",
        job_id="1",
        job=Job(
            status="Running",
            start_time=datetime(year=2020, month=10, day=27),
        ),
    )
    update_event.update_job(
        real_id="9",
        stage_id="0",
        step_id="0",
        job_id="0",
        job=Job(
            status="Running",
            start_time=datetime(year=2020, month=10, day=27),
        ),
    )

    snapshot.merge_event(update_event)

    assert snapshot.get_status() == "running"

    assert snapshot.get_job(real_id="1", stage_id="0", step_id="0", job_id="0") == Job(
        status="Finished",
        start_time=datetime(year=2020, month=10, day=27),
        end_time=datetime(year=2020, month=10, day=28),
        data={"memory": 1000},
        error=None,
        name="job0",
        stderr=None,
        stdout=None,
    )

    assert snapshot.get_job(real_id="1", stage_id="0", step_id="0", job_id="1") == Job(
        status="Running",
        start_time=datetime(year=2020, month=10, day=27),
        end_time=None,
        data={},
        error=None,
        name="job1",
        stderr=None,
        stdout=None,
    )

    assert (
        snapshot.get_job(real_id="9", stage_id="0", step_id="0", job_id="0").status
        == "Running"
    )
    assert snapshot.get_job(real_id="9", stage_id="0", step_id="0", job_id="0") == Job(
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
            "/ert/ee/0/real/1111/stage/2stage_id/step/asd123ASD/job/0",
            {"real": "1111", "stage": "2stage_id", "step": "asd123ASD", "job": "0"},
        ),
        (
            "/ert/ee/0/real/1111/stage/2stage_id/step/asd123ASD",
            {"real": "1111", "stage": "2stage_id", "step": "asd123ASD", "job": None},
        ),
        (
            "/ert/ee/0/real/1111/stage/2stage_id",
            {"real": "1111", "stage": "2stage_id", "step": None, "job": None},
        ),
        (
            "/ert/ee/0/real/1111",
            {"real": "1111", "stage": None, "step": None, "job": None},
        ),
        (
            "/ert/ee/0",
            {"real": None, "stage": None, "step": None, "job": None},
        ),
    ],
)
def test_source_get_ids(source_string, expected_ids):

    assert tool.get_real_id(source_string) == expected_ids["real"]
    assert tool.get_stage_id(source_string) == expected_ids["stage"]
    assert tool.get_step_id(source_string) == expected_ids["step"]
    assert tool.get_job_id(source_string) == expected_ids["job"]


def test_commands_to_and_from_dict():
    pause_command = command.create_command_pause()
    terminate_command = command.create_command_terminate()

    assert pause_command == command.create_command_from_dict(pause_command.to_dict())
    assert terminate_command == command.create_command_from_dict(
        terminate_command.to_dict()
    )
