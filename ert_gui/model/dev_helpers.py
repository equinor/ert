from ert_gui.model.node import Node, NodeType
from ert_shared.ensemble_evaluator.entity.snapshot import (
    PartialSnapshot,
    _SnapshotDict,
    _ForwardModel,
    _Job,
    _Realization,
    _Stage,
    _Step,
)


def create_partial_snapshot(real_id, job_status) -> PartialSnapshot:
    partial = PartialSnapshot()
    partial.update_real(real_id, status=job_status)
    partial.update_job(real_id, "0", "0", "0", status=job_status)
    return partial


def create_snapshot() -> _SnapshotDict:
    snapshot = _SnapshotDict(
        status="Unknown",
        reals={
            "0": _Realization(
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
        },
        forward_model=_ForwardModel(step_definitions={}),
    )

    return snapshot
