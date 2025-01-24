import json
from collections import defaultdict
from datetime import datetime
from functools import partial
from textwrap import dedent
from unittest.mock import MagicMock, patch

import pytest
from websockets.sync.client import ClientConnection

import everest
from ert.ensemble_evaluator import (
    EndEvent,
    FullSnapshotEvent,
    SnapshotUpdateEvent,
    state,
)
from ert.ensemble_evaluator.snapshot import EnsembleSnapshotMetadata
from everest.bin.utils import run_detached_monitor
from tests.ert import SnapshotBuilder

METADATA = EnsembleSnapshotMetadata(
    aggr_fm_step_status_colors=defaultdict(dict),
    real_status_colors={},
    sorted_real_ids=[],
    sorted_fm_step_ids=defaultdict(list),
)


from fastapi.encoders import jsonable_encoder


@pytest.fixture
def full_snapshot_event():
    event = FullSnapshotEvent(
        snapshot=(
            SnapshotBuilder(metadata=METADATA)
            .add_fm_step(
                fm_step_id="0",
                index="0",
                name="fm_step_0",
                status=state.FORWARD_MODEL_STATE_START,
                current_memory_usage="500",
                max_memory_usage="1000",
                stdout="job_fm_step_0.stdout",
                stderr="job_fm_step_0.stderr",
                start_time=datetime(1999, 1, 1),
            )
            .build(
                real_ids=["0", "1"],
                status=state.REALIZATION_STATE_PENDING,
                start_time=datetime(1999, 1, 1),
                exec_hosts="12121.121",
                message="Some message",
            )
        ),
        iteration_label="Foo",
        total_iterations=1,
        progress=0.25,
        realization_count=4,
        status_count={"Finished": 0, "Pending": 1, "Unknown": 0},
        iteration=0,
    )
    yield json.dumps(jsonable_encoder(event))


@pytest.fixture
def snapshot_update_event():
    event = SnapshotUpdateEvent(
        snapshot=SnapshotBuilder(metadata=METADATA)
        .add_fm_step(
            fm_step_id="0",
            name=None,
            index="0",
            status=state.FORWARD_MODEL_STATE_FINISHED,
            end_time=datetime(2019, 1, 1),
        )
        .build(
            real_ids=["1"],
            status=state.REALIZATION_STATE_FINISHED,
        ),
        iteration_label="Foo",
        total_iterations=1,
        progress=0.5,
        realization_count=4,
        status_count={"Finished": 1, "Running": 0, "Unknown": 0},
        iteration=0,
    )
    yield json.dumps(jsonable_encoder(event))


def test_monitor(monkeypatch, full_snapshot_event, snapshot_update_event, capsys):
    server_mock = MagicMock()
    connection_mock = MagicMock(spec=ClientConnection)
    connection_mock.recv.side_effect = [
        full_snapshot_event,
        snapshot_update_event,
        json.dumps(jsonable_encoder(EndEvent(failed=False, msg="Experiment complete"))),
    ]
    server_mock.return_value.__enter__.return_value = connection_mock
    monkeypatch.setattr(everest.detached, "_query_server", MagicMock(return_value={}))
    monkeypatch.setattr(everest.detached, "connect", server_mock)
    monkeypatch.setattr(everest.detached, "ssl", MagicMock())
    patched = partial(everest.detached.start_monitor, polling_interval=0.1)
    with patch("everest.bin.utils.start_monitor", patched):
        run_detached_monitor(("some/url", None, None), "output", True)
    captured = capsys.readouterr()
    expected = dedent("""
    ===================== Running forward models (Batch #0) ======================

      Waiting: 0 | Pending: 0 | Running: 0 | Complete: 0 | Failed: 0

      fm_step_0: 0/1/0 | Finished: 1

    Experiment complete
    """)
    assert captured.out == expected.lstrip()
