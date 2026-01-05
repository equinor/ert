import copy
from datetime import datetime as dt

import pytest

from ert.ensemble_evaluator.snapshot import (
    EnsembleSnapshot,
    FMStepSnapshot,
    RealizationSnapshot,
)
from ert.ensemble_evaluator.state import (
    ENSEMBLE_STATE_STARTED,
    FORWARD_MODEL_STATE_FINISHED,
    FORWARD_MODEL_STATE_START,
    REALIZATION_STATE_FAILED,
    REALIZATION_STATE_RUNNING,
    REALIZATION_STATE_UNKNOWN,
)
from tests.ert import SnapshotBuilder


@pytest.fixture
def full_snapshot() -> EnsembleSnapshot:
    real = RealizationSnapshot(
        status=REALIZATION_STATE_RUNNING,
        active=True,
        exec_hosts="COMP-01",
        fm_steps={
            "0": FMStepSnapshot(
                start_time=dt.now(),
                end_time=dt.now(),
                name="poly_eval",
                index="0",
                status=FORWARD_MODEL_STATE_START,
                error="error",
                stdout="std_out_file",
                stderr="std_err_file",
                current_memory_usage=123,
                max_memory_usage=312,
            ),
            "1": FMStepSnapshot(
                start_time=dt.now(),
                end_time=dt.now(),
                name="poly_postval",
                index="1",
                status=FORWARD_MODEL_STATE_START,
                error="error",
                stdout="std_out_file",
                stderr="std_err_file",
                current_memory_usage=123,
                max_memory_usage=312,
            ),
            "2": FMStepSnapshot(
                start_time=dt.now(),
                end_time=None,
                name="poly_post_mortem",
                index="2",
                status=FORWARD_MODEL_STATE_START,
                error="error",
                stdout="std_out_file",
                stderr="std_err_file",
                current_memory_usage=123,
                max_memory_usage=312,
            ),
            "3": FMStepSnapshot(
                start_time=dt.now(),
                end_time=None,
                name="poly_not_started",
                index="3",
                status=FORWARD_MODEL_STATE_START,
                error="error",
                stdout="std_out_file",
                stderr="std_err_file",
                current_memory_usage=123,
                max_memory_usage=312,
            ),
        },
    )
    snapshot = EnsembleSnapshot()
    for i in range(100):
        snapshot.add_realization(str(i), copy.deepcopy(real))

    return snapshot


@pytest.fixture
def fail_snapshot() -> EnsembleSnapshot:
    real = RealizationSnapshot(
        status=REALIZATION_STATE_FAILED,
        active=True,
        fm_steps={
            "0": FMStepSnapshot(
                start_time=dt.now(),
                end_time=dt.now(),
                name="poly_eval",
                index="0",
                status=FORWARD_MODEL_STATE_FINISHED,
                error="error",
                stdout="std_out_file",
                stderr="std_err_file",
                current_memory_usage=123,
                max_memory_usage=312,
            )
        },
    )
    snapshot = EnsembleSnapshot()
    snapshot._ensemble_state = ENSEMBLE_STATE_STARTED

    for i in range(1):
        snapshot.add_realization(str(i), copy.deepcopy(real))

    return snapshot


@pytest.fixture
def large_snapshot() -> EnsembleSnapshot:
    builder = SnapshotBuilder()
    for i in range(150):
        builder.add_fm_step(
            fm_step_id=str(i),
            index=str(i),
            name=f"job_{i}",
            current_memory_usage="500",
            max_memory_usage="1000",
            status=FORWARD_MODEL_STATE_START,
            stdout=f"job_{i}.stdout",
            stderr=f"job_{i}.stderr",
            start_time=dt(1999, 1, 1),
            end_time=dt(2019, 1, 1),
        )
    real_ids = [str(i) for i in range(150)]
    return builder.build(real_ids, REALIZATION_STATE_UNKNOWN)
