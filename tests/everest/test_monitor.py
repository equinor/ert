import json
import os
import shutil
import string
from collections import defaultdict
from datetime import UTC, datetime
from unittest.mock import MagicMock

import pytest
from fastapi.encoders import jsonable_encoder

from ert.ensemble_evaluator import (
    EndEvent,
    FullSnapshotEvent,
    SnapshotUpdateEvent,
    state,
)
from ert.ensemble_evaluator.snapshot import EnsembleSnapshotMetadata
from ert.resources import all_shell_script_fm_steps
from ert.run_models.event import EverestBatchResultEvent, status_event_from_json
from ert.services import ErtClient
from everest.bin.utils import run_detached_monitor, run_empty_detached_monitor
from everest.detached.client import start_monitor
from everest.strings import SIM_PROGRESS_ID
from tests.ert.utils import SnapshotBuilder

METADATA = EnsembleSnapshotMetadata(
    fm_step_status=defaultdict(dict),
    real_status={},
    sorted_real_ids=[],
    sorted_fm_step_ids=defaultdict(list),
)


@pytest.fixture(autouse=True)
def fixed_terminal_width(monkeypatch):
    monkeypatch.setattr(
        shutil, "get_terminal_size", lambda *args, **kwargs: os.terminal_size((60, 24))
    )


@pytest.fixture
def monitor_client():
    return MagicMock(spec=ErtClient)


def test_that_monitor_delivers_events_after_end_event(monitor_client):
    events = [EndEvent(failed=False, msg="first"), EndEvent(failed=True, msg="last")]
    monitor_client.iter_events.return_value = (event for event in events)
    callback = MagicMock()

    start_monitor(monitor_client, callback, "experiment", polling_interval=0.2)

    monitor_client.iter_events.assert_called_once_with(
        "experiment", refresh_interval=0.2
    )
    assert [call.args[0][SIM_PROGRESS_ID] for call in callback.call_args_list] == events


def test_that_empty_monitor_consumes_all_events_without_output(monitor_client, capsys):
    consumed = []

    def iter_events():
        for message in ["first", "last"]:
            yield EndEvent(failed=False, msg=message)
            consumed.append(message)

    monitor_client.iter_events.return_value = iter_events()

    run_empty_detached_monitor(monitor_client, "experiment")

    assert consumed == ["first", "last"]
    assert not capsys.readouterr().out


@pytest.mark.parametrize("exception_type", [RuntimeError, KeyboardInterrupt])
def test_that_callback_exception_propagates_from_monitor(
    monitor_client, exception_type
):
    monitor_client.iter_events.return_value = iter(
        [EndEvent(failed=False, msg="completed")]
    )
    callback = MagicMock(side_effect=exception_type)

    with pytest.raises(exception_type):
        start_monitor(monitor_client, callback, "experiment")


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
        start_time=datetime(1999, 1, 1, tzinfo=UTC),
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
            start_time=datetime(1999, 1, 1, tzinfo=UTC),
        )
    event = FullSnapshotEvent(
        snapshot=snapshot.build(
            real_ids=["0", "1"],
            status=state.REALIZATION_STATE_PENDING,
            start_time=datetime(1999, 1, 1, tzinfo=UTC),
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
    return json.dumps(jsonable_encoder(event))


@pytest.fixture
def snapshot_update_event():
    event = SnapshotUpdateEvent(
        snapshot=SnapshotBuilder(metadata=METADATA)
        .add_fm_step(
            fm_step_id="0",
            name=None,
            index="0",
            status=state.FORWARD_MODEL_STATE_FINISHED,
            end_time=datetime(2019, 1, 1, tzinfo=UTC),
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
    return json.dumps(jsonable_encoder(event))


@pytest.fixture
def everest_batch_result_event():
    event = EverestBatchResultEvent(
        batch=0,
        everest_event="OPTIMIZATION_RESULT",
        result_type="GradientResult",
        results={"dummy": "ignored"},
        failures={0: [-1], 1: [0, 1, 2]},
    )
    return json.dumps(jsonable_encoder(event))


@pytest.fixture
def snapshot_update_failure_event():
    event = SnapshotUpdateEvent(
        snapshot=SnapshotBuilder(metadata=METADATA)
        .add_fm_step(
            fm_step_id="0",
            name=None,
            index="0",
            status=state.FORWARD_MODEL_STATE_FAILURE,
            end_time=datetime(2019, 1, 1, tzinfo=UTC),
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
    return json.dumps(jsonable_encoder(event))


@pytest.fixture
def snapshot_update_event_with_fm_message():
    event = SnapshotUpdateEvent(
        snapshot=SnapshotBuilder(metadata=METADATA)
        .add_fm_step(
            fm_step_id="0",
            name=None,
            index="0",
            status=state.FORWARD_MODEL_STATE_FINISHED,
            end_time=datetime(2019, 1, 1, tzinfo=UTC),
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
    return json.dumps(jsonable_encoder(event))


@pytest.mark.slow
def test_that_the_monitor_shows_failed_jobs(
    monitor_client, full_snapshot_event, snapshot_update_failure_event, capsys
):
    monitor_client.iter_events.return_value = (
        status_event_from_json(message)
        for message in [
            full_snapshot_event,
            snapshot_update_failure_event,
            json.dumps(jsonable_encoder(EndEvent(failed=True, msg="Failed"))),
        ]
    )
    run_detached_monitor(
        monitor_client,
        experiment_id="test-experiment-id",
    )
    captured = capsys.readouterr()
    expected = [
        "============ Running forward models (Batch #0) =============\n",
        "  Waiting: 0 | Pending: 0 | Running: 0 | Finished: 0 | Failed: 1\n",
        (
            "  fm_step_0: 1/0/1"
            "  fm_step_0: The run is cancelled due to reaching MAX_RUNTIME\n"
        ),
    ]
    # Ignore whitespace
    output = captured.out.translate({ord(c): None for c in string.whitespace})
    assert output.startswith(
        "".join(expected).translate({ord(c): None for c in string.whitespace})
    )


@pytest.mark.slow
def test_that_the_monitor_shows_running_jobs(
    monitor_client, full_snapshot_event, snapshot_update_event, capsys
):
    monitor_client.iter_events.return_value = (
        status_event_from_json(message)
        for message in [
            full_snapshot_event,
            snapshot_update_event,
            json.dumps(
                jsonable_encoder(EndEvent(failed=False, msg="Experiment completed"))
            ),
        ]
    )
    run_detached_monitor(monitor_client, experiment_id="test-experiment-id")
    captured = capsys.readouterr()
    expected = [
        "============ Running forward models (Batch #0) =============\n",
        "  Waiting: 0 | Pending: 0 | Running: 0 | Finished: 1 | Failed: 0\n",
        "  fm_step_0: 1/1/0\n",
    ]
    expected.extend([f"{name}: 2/0/0" for name in all_shell_script_fm_steps])
    # Ignore whitespace
    assert captured.out.translate({ord(c): None for c in string.whitespace}) == "".join(
        expected
    ).translate({ord(c): None for c in string.whitespace})


@pytest.mark.slow
def test_that_a_forward_model_message_reaches_the_cli(
    monitor_client, full_snapshot_event, snapshot_update_event_with_fm_message, capsys
):
    monitor_client.iter_events.return_value = (
        status_event_from_json(message)
        for message in [
            full_snapshot_event,
            snapshot_update_event_with_fm_message,
            json.dumps(jsonable_encoder(EndEvent(failed=True, msg="Failed"))),
        ]
    )
    run_detached_monitor(
        monitor_client,
        experiment_id="test-experiment-id",
    )
    captured = capsys.readouterr()

    expected = [
        "============ Running forward models (Batch #0) =============\n",
        "  Waiting: 0 | Pending: 0 | Running: 0 | Finished: 1 | Failed: 0\n",
        "  fm_step_0: 1/1/0\n",
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


@pytest.mark.slow
def test_that_a_failed_everest_batch_result_event_is_shown(
    monitor_client, everest_batch_result_event, capsys
):
    monitor_client.iter_events.return_value = (
        status_event_from_json(message)
        for message in [
            everest_batch_result_event,
            json.dumps(jsonable_encoder(EndEvent(failed=True, msg="Failed"))),
        ]
    )
    run_detached_monitor(monitor_client, experiment_id="test-run-id")
    captured = capsys.readouterr()
    expected = [
        "============= Optimization progress (Batch #0) =============\n",
        "  Failed function evaluation for realization: 0\n",
        "  Failed perturbations for realization 1: 0-2\n",
    ]
    # Ignore whitespace
    output = captured.out.translate({ord(c): None for c in string.whitespace})
    assert output.startswith(
        "".join(expected).translate({ord(c): None for c in string.whitespace})
    )
