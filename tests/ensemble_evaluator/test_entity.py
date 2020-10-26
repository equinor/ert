from ert_shared.ensemble_evaluator.entity.snapshot import (
    SnapshotBuilder,
    PartialSnapshot,
    Snapshot,
)
from ert_shared.ensemble_evaluator.entity.tool import (
    get_real_id,
    get_stage_id,
    get_step_id,
    get_job_id,
)


def _dict_equal(d1, d2):
    if set(d1.keys()) != set(d2.keys()):
        return False

    for k in d1:
        if type(d1[k]) is dict:
            if not _dict_equal(d1[k], d2[k]):
                return False
        else:
            if d1[k] != d2[k]:
                return False
    return True


_REALIZATION_INDEXES = ["0", "1", "3", "4", "5", "9"]


def _create_snapshot():
    return (
        SnapshotBuilder()
        .add_stage(stage_id="0", status="unknown")
        .add_step(stage_id="0", step_id="0", status="unknown")
        .add_job(stage_id="0", step_id="0", job_id="0", data={}, status="unknown")
        .add_job(
            stage_id="0",
            step_id="0",
            job_id="1",
            data={},
            status="unknown",
            depends=[("0", "0", "0")],
        )
        .add_job(
            stage_id="0",
            step_id="0",
            job_id="2",
            data={},
            status="unknown",
            depends=[("0", "0", "0")],
        )
        .add_job(
            stage_id="0",
            step_id="0",
            job_id="3",
            data={},
            status="unknown",
            depends=[("0", "0", "1"), ("0", "0", "2")],
        )
        .build(_REALIZATION_INDEXES, status="unknown")
    )


def test_snapshot_merge():
    snapshot = _create_snapshot()

    update_event = PartialSnapshot()
    update_event.update_status(status="running")

    snapshot.merge_event(update_event)

    assert snapshot.get_status() == "running"

    update_event = PartialSnapshot()
    update_event.update_job(
        real_id="1",
        stage_id="0",
        step_id="0",
        job_id="0",
        status="success",
        data={"memory": 1000},
    )
    update_event.update_job(
        real_id="1", stage_id="0", step_id="0", job_id="1", status="running"
    )
    update_event.update_job(
        real_id="9", stage_id="0", step_id="0", job_id="0", status="running"
    )

    snapshot.merge_event(update_event)

    assert snapshot.get_status() == "running"

    assert _dict_equal(
        snapshot.get_job(real_id="1", stage_id="0", step_id="0", job_id="0"),
        {"status": "success", "data": {"memory": 1000}},
    )
    assert snapshot.get_job(real_id="1", stage_id="0", step_id="0", job_id="1") == {
        "status": "running",
        "data": {},
    }

    assert (
        snapshot.get_job(real_id="9", stage_id="0", step_id="0", job_id="0")["status"]
        == "running"
    )
    assert snapshot.get_job(real_id="9", stage_id="0", step_id="0", job_id="0") == {
        "status": "running",
        "data": {},
    }


def test_source_get_id():
    source = "/ert/ee/0/real/1111/stage/2opop/step/asd123ASD/job/0"

    assert get_real_id(source) == "1111"
    assert get_stage_id(source) == "2opop"
    assert get_step_id(source) == "asd123ASD"
    assert get_job_id(source) == "0"
