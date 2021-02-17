from datetime import datetime
from unittest.mock import Mock

import ert_shared.ensemble_evaluator.entity.identifiers as ids
import pytest
from ert_shared.ensemble_evaluator.entity.snapshot import Snapshot, SnapshotBuilder
from ert_shared.status.entity.state import (
    JOB_STATE_START,
    REALIZATION_STATE_UNKNOWN,
    STAGE_STATE_UNKNOWN,
    STEP_STATE_START,
)


@pytest.fixture()
def full_snapshot() -> Snapshot:
    builder = (
        SnapshotBuilder()
        .add_stage(stage_id="0", status=STAGE_STATE_UNKNOWN)
        .add_step(stage_id="0", step_id="0", status=STEP_STATE_START)
    )
    for i in range(0, 150):
        builder.add_job(
            stage_id="0",
            step_id="0",
            job_id=str(i),
            name=f"job_{i}",
            data={ids.MAX_MEMORY_USAGE: 1000, ids.CURRENT_MEMORY_USAGE: 500},
            status=JOB_STATE_START,
            stdout=f"job_{i}.stdout",
            stderr=f"job_{i}.stderr",
            start_time=datetime(1999, 1, 1).isoformat(),
            end_time=datetime(2019, 1, 1).isoformat(),
        )
    real_ids = [str(i) for i in range(0, 150)]
    return builder.build(real_ids, REALIZATION_STATE_UNKNOWN)


@pytest.fixture
def runmodel() -> Mock:
    brm = Mock()
    brm.get_runtime = Mock(return_value=100)
    brm.hasRunFailed = Mock(return_value=False)
    brm.getFailMessage = Mock(return_value="")
    brm.support_restart = True
    return brm


@pytest.fixture
def active_realizations() -> Mock:
    active_reals = Mock()
    active_reals.count = Mock(return_value=10)
    return active_reals
