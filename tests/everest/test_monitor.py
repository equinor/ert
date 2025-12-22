import json
import string
from collections import defaultdict
from datetime import datetime
from functools import partial
from unittest.mock import MagicMock, patch

import pytest
from fastapi.encoders import jsonable_encoder
from websockets.sync.client import ClientConnection

import everest
from ert.ensemble_evaluator import (
    EndEvent,
    FullSnapshotEvent,
    SnapshotUpdateEvent,
    state,
)
from ert.ensemble_evaluator.snapshot import EnsembleSnapshotMetadata
from ert.resources import all_shell_script_fm_steps
from everest.bin.utils import run_detached_monitor
from tests.ert import SnapshotBuilder

METADATA = EnsembleSnapshotMetadata(
    fm_step_status=defaultdict(dict),
    real_status={},
    sorted_real_ids=[],
    sorted_fm_step_ids=defaultdict(list),
)


@pytest.fixture
def full_snapshot_event():
    snapshot = SnapshotBuilder(metadata=METADATA)
    snapshot.add_fm_step(
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
    for i, command in enumerate(all_shell_script_fm_steps):
        snapshot.add_fm_step(
            fm_step_id=str(i + 1),
            index=str(i + 1),
            name=command,
            status=state.FORWARD_MODEL_STATE_START,
            current_memory_usage="500",
            max_memory_usage="1000",
            stdout=None,
            stderr=None,
            start_time=datetime(1999, 1, 1),
        )
    event = FullSnapshotEvent(
        snapshot=snapshot.build(
            real_ids=["0", "1"],
            status=state.REALIZATION_STATE_PENDING,
            start_time=datetime(1999, 1, 1),
            exec_hosts="12121.121",
            message="",
        ),
        iteration_label="Foo",
        total_iterations=1,
        progress=0.25,
        realization_count=4,
        status_count={
            "Finished": 0,
            "Pending": len(all_shell_script_fm_steps),
            "Unknown": 0,
        },
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


@pytest.fixture
def snapshot_update_failure_event():
    event = SnapshotUpdateEvent(
        snapshot=SnapshotBuilder(metadata=METADATA)
        .add_fm_step(
            fm_step_id="0",
            name=None,
            index="0",
            status=state.FORWARD_MODEL_STATE_FAILURE,
            end_time=datetime(2019, 1, 1),
            error="The run is cancelled due to reaching MAX_RUNTIME",
        )
        .build(
            real_ids=["1"],
            status=state.REALIZATION_STATE_FAILED,
        ),
        iteration_label="Foo",
        total_iterations=1,
        progress=0.5,
        realization_count=4,
        status_count={"Finished": 0, "Running": 0, "Unknown": 0, "Failed": 1},
        iteration=0,
    )
    yield json.dumps(jsonable_encoder(event))


@pytest.fixture
def snapshot_update_event_with_fm_message():
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
            message="Something went wrong!",
        ),
        iteration_label="Foo",
        total_iterations=1,
        progress=0.5,
        realization_count=4,
        status_count={"Finished": 1, "Running": 0, "Unknown": 0},
        iteration=0,
    )
    yield json.dumps(jsonable_encoder(event))


@pytest.mark.integration_test
def test_failed_jobs_monitor(
    monkeypatch, full_snapshot_event, snapshot_update_failure_event, capsys
):
    server_mock = MagicMock()
    connection_mock = MagicMock(spec=ClientConnection)
    connection_mock.recv.side_effect = [
        full_snapshot_event,
        snapshot_update_failure_event,
        json.dumps(jsonable_encoder(EndEvent(failed=True, msg="Failed"))),
    ]
    server_mock.return_value.__enter__.return_value = connection_mock
    monkeypatch.setattr(everest.detached.client, "connect", server_mock)
    monkeypatch.setattr(everest.detached.client, "ssl", MagicMock())
    partial(everest.detached.start_monitor, polling_interval=0.1)
    run_detached_monitor(("some/url", "cert", ("username", "password")))
    captured = capsys.readouterr()
    expected = [
        "===================== Running forward models (Batch #0) ======================\n",  # noqa: E501
        "  Waiting: 0 | Pending: 0 | Running: 0 | Finished: 0 | Failed: 1\n",
        (
            "  fm_step_0: 1/0/1 | Failed: 1"
            "  fm_step_0: Failed: "
            "The run is cancelled due to reaching MAX_RUNTIME, realizations: 1\n"
        ),
    ]
    # Ignore whitespace
    output = captured.out.translate({ord(c): None for c in string.whitespace})
    assert output.startswith(
        "".join(expected).translate({ord(c): None for c in string.whitespace})
    )


@pytest.mark.integration_test
def test_monitor(monkeypatch, full_snapshot_event, snapshot_update_event, capsys):
    server_mock = MagicMock()
    connection_mock = MagicMock(spec=ClientConnection)
    connection_mock.recv.side_effect = [
        full_snapshot_event,
        snapshot_update_event,
        json.dumps(
            jsonable_encoder(EndEvent(failed=False, msg="Experiment completed"))
        ),
    ]
    server_mock.return_value.__enter__.return_value = connection_mock
    monkeypatch.setattr(everest.detached.client, "connect", server_mock)
    monkeypatch.setattr(everest.detached.client, "ssl", MagicMock())
    patched = partial(everest.detached.start_monitor, polling_interval=0.1)
    with patch("everest.bin.utils.start_monitor", patched):
        run_detached_monitor(("some/url", "cert", ("username", "password")))
    captured = capsys.readouterr()
    expected = [
        "===================== Running forward models (Batch #0) ======================\n",  # noqa: E501
        "  Waiting: 0 | Pending: 0 | Running: 0 | Finished: 1 | Failed: 0\n",
        "  fm_step_0: 1/1/0 | Finished: 1\n",
    ]
    expected.extend([f"{name}: 2/0/0" for name in all_shell_script_fm_steps])
    # Ignore whitespace
    assert captured.out.translate({ord(c): None for c in string.whitespace}) == "".join(
        expected
    ).translate({ord(c): None for c in string.whitespace})


@pytest.mark.integration_test
def test_forward_model_message_reaches_the_cli(
    monkeypatch, full_snapshot_event, snapshot_update_event_with_fm_message, capsys
):
    server_mock = MagicMock()
    connection_mock = MagicMock(spec=ClientConnection)
    connection_mock.recv.side_effect = [
        full_snapshot_event,
        snapshot_update_event_with_fm_message,
        json.dumps(jsonable_encoder(EndEvent(failed=True, msg="Failed"))),
    ]
    server_mock.return_value.__enter__.return_value = connection_mock
    monkeypatch.setattr(everest.detached.client, "connect", server_mock)
    monkeypatch.setattr(everest.detached.client, "ssl", MagicMock())
    partial(everest.detached.start_monitor, polling_interval=0.1)
    run_detached_monitor(("some/url", "cert", ("username", "password")))
    captured = capsys.readouterr()

    expected = [
        "===================== Running forward models (Batch #0) ======================\n",  # noqa: E501
        "  Waiting: 0 | Pending: 0 | Running: 0 | Finished: 1 | Failed: 0\n",
        "  fm_step_0: 1/1/0 | Finished: 1\n",
    ]
    expected.append("Something went wrong!\n")
    expected.extend(
        [
            f"{name}: 2/0/0\n Something went wrong!\n"
            for name in all_shell_script_fm_steps
        ]
    )

    # Ignore whitespace
    assert captured.out.translate({ord(c): None for c in string.whitespace}) == "".join(
        expected
    ).translate({ord(c): None for c in string.whitespace})
