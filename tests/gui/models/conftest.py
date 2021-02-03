from ert_shared.status.entity.state import JOB_STATE_FINISHED
import pytest
from ert_shared.ensemble_evaluator.entity.snapshot import (
    PartialSnapshot, Snapshot,
    _ForwardModel,
    _Job,
    _Realization,
    _SnapshotDict,
    _Stage,
    _Step,
)
import copy


def partial_snapshot(snapshot) -> PartialSnapshot:
    partial = PartialSnapshot(snapshot)
    partial.update_real("0", status=JOB_STATE_FINISHED)
    partial.update_job("0", "0", "0", "0", status=JOB_STATE_FINISHED)
    return partial


@pytest.fixture()
def full_snapshot() -> Snapshot:
    real = _Realization(
        status="Unknown",
        active=True,
        stages={
            "0": _Stage(
                status="",
                steps={
                    "0": _Step(
                        status="",
                        jobs={
                            "0": _Job(
                                start_time=str(123),
                                end_time=str(123),
                                name="poly_eval",
                                status="Unknown",
                                error="error",
                                stdout="std_out_file",
                                stderr="std_err_file",
                                data={
                                    "current_memory_usage": "123",
                                    "max_memory_usage": "312",
                                },
                            ),
                            "1": _Job(
                                start_time=str(123),
                                end_time=str(123),
                                name="poly_postval",
                                status="Workin",
                                error="error",
                                stdout="std_out_file",
                                stderr="std_err_file",
                                data={
                                    "current_memory_usage": "123",
                                    "max_memory_usage": "312",
                                },
                            ),
                        },
                    )
                },
            )
        },
    )
    snapshot = _SnapshotDict(
        status="Unknown",
        reals={},
        forward_model=_ForwardModel(step_definitions={}),
    )
    for i in range(0, 100):
        snapshot.reals[str(i)] = copy.deepcopy(real)

    return Snapshot(snapshot.dict())
