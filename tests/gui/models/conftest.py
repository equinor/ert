from ert_shared.ensemble_evaluator.entity.identifiers import (
    CURRENT_MEMORY_USAGE,
    MAX_MEMORY_USAGE,
)
from ert_shared.status.entity.state import (
    ENSEMBLE_STATE_STARTED,
    JOB_STATE_FINISHED,
    JOB_STATE_START,
    REALIZATION_STATE_UNKNOWN,
)
import pytest
from ert_shared.ensemble_evaluator.entity.snapshot import (
    PartialSnapshot,
    Snapshot,
    _ForwardModel,
    _Job,
    _Realization,
    _SnapshotDict,
    _Stage,
    _Step,
)
import copy
import datetime


def partial_snapshot(snapshot) -> PartialSnapshot:
    partial = PartialSnapshot(snapshot)
    partial.update_real("0", status=JOB_STATE_FINISHED)
    partial.update_job("0", "0", "0", "0", status=JOB_STATE_FINISHED)
    return partial


@pytest.fixture()
def full_snapshot() -> Snapshot:
    real = _Realization(
        status=REALIZATION_STATE_UNKNOWN,
        active=True,
        stages={
            "0": _Stage(
                status="",
                steps={
                    "0": _Step(
                        status="",
                        jobs={
                            "0": _Job(
                                start_time=datetime.datetime.now(),
                                end_time=datetime.datetime.now(),
                                name="poly_eval",
                                status=JOB_STATE_START,
                                error="error",
                                stdout="std_out_file",
                                stderr="std_err_file",
                                data={
                                    CURRENT_MEMORY_USAGE: "123",
                                    MAX_MEMORY_USAGE: "312",
                                },
                            ),
                            "1": _Job(
                                start_time=datetime.datetime.now(),
                                end_time=datetime.datetime.now(),
                                name="poly_postval",
                                status=JOB_STATE_START,
                                error="error",
                                stdout="std_out_file",
                                stderr="std_err_file",
                                data={
                                    CURRENT_MEMORY_USAGE: "123",
                                    MAX_MEMORY_USAGE: "312",
                                },
                            ),
                        },
                    )
                },
            )
        },
    )
    snapshot = _SnapshotDict(
        status=ENSEMBLE_STATE_STARTED,
        reals={},
        forward_model=_ForwardModel(step_definitions={}),
    )
    for i in range(0, 100):
        snapshot.reals[str(i)] = copy.deepcopy(real)

    return Snapshot(snapshot.dict())
