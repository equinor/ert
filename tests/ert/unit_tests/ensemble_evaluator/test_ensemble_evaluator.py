import asyncio
import datetime
from functools import partial

import pytest
import zmq.asyncio
from hypothesis import given
from hypothesis import strategies as st
from pydantic import ValidationError

from _ert.events import (
    EESnapshot,
    EESnapshotUpdate,
    EnsembleStarted,
    EnsembleSucceeded,
    Event,
    ForwardModelStepFailure,
    ForwardModelStepRunning,
    ForwardModelStepSuccess,
    Id,
    RealizationSuccess,
    event_from_dict,
    event_to_json,
)
from _ert.forward_model_runner.client import (
    CONNECT_MSG,
    DISCONNECT_MSG,
    Client,
)
from ert.ensemble_evaluator import (
    EnsembleEvaluator,
    EnsembleSnapshot,
    FMStepSnapshot,
)
from ert.ensemble_evaluator._ensemble import LegacyEnsemble
from ert.ensemble_evaluator.evaluator import detect_overspent_cpu
from ert.ensemble_evaluator.state import (
    ENSEMBLE_STATE_STARTED,
    ENSEMBLE_STATE_STOPPED,
    ENSEMBLE_STATE_UNKNOWN,
    FORWARD_MODEL_STATE_FAILURE,
    FORWARD_MODEL_STATE_FINISHED,
    FORWARD_MODEL_STATE_INIT,
)
from ert.scheduler import JobState

from .ensemble_evaluator_utils import TestEnsemble


@pytest.mark.parametrize(
    "task, error_msg",
    [
        ("_batch_events_into_buffer", "Batcher failed!"),
        ("_process_event_buffer", "Batch processing failed!"),
    ],
)
async def test_when_task_fails_evaluator_raises_exception(
    task, error_msg, make_ee_config, monkeypatch
):
    async def mock_failure(message, *args, **kwargs):
        raise RuntimeError(message)

    evaluator = EnsembleEvaluator(
        TestEnsemble(0, 2, 2, id_="0"), make_ee_config(use_token=False)
    )

    monkeypatch.setattr(
        EnsembleEvaluator,
        task,
        partial(mock_failure, error_msg),
    )
    with pytest.raises(RuntimeError, match=error_msg):
        await evaluator.run_and_get_successful_realizations()


async def test_evaluator_raises_on_invalid_dispatch_event(evaluator_to_use):
    (evaluator, _) = evaluator_to_use

    with pytest.raises(ValidationError):
        await evaluator.handle_dispatch(b"dispatcher-1", b"This is not an event!!")


async def test_evaluator_handles_dispatchers_connected(
    evaluator_to_use,
):
    (evaluator, _) = evaluator_to_use

    await evaluator.handle_dispatch(b"dispatcher-1", CONNECT_MSG)
    await evaluator.handle_dispatch(b"dispatcher-2", CONNECT_MSG)
    assert not evaluator._dispatchers_empty.is_set()
    assert evaluator._dispatchers_connected == {b"dispatcher-1", b"dispatcher-2"}
    await evaluator.handle_dispatch(b"dispatcher-1", DISCONNECT_MSG)
    await evaluator.handle_dispatch(b"dispatcher-2", DISCONNECT_MSG)
    assert evaluator._dispatchers_empty.is_set()


async def test_evaluator_raises_on_start_with_address_in_use(make_ee_config):
    ee_config = make_ee_config(use_ipc_protocol=False)
    ctx = zmq.asyncio.Context()
    socket = ctx.socket(zmq.ROUTER)
    try:
        ee_config.router_port = socket.bind_to_random_port(
            "tcp://*",
            min_port=ee_config.min_port,
            max_port=ee_config.max_port,
        )
        ee_config.min_port = ee_config.router_port
        ee_config.max_port = ee_config.router_port + 1
        evaluator = EnsembleEvaluator(TestEnsemble(0, 2, 2, id_="0"), ee_config)
        with pytest.raises(
            zmq.error.ZMQBindError, match="Could not bind socket to random port"
        ):
            await evaluator.run_and_get_successful_realizations()
    finally:
        socket.close()
        ctx.destroy(linger=0)


async def test_no_config_raises_valueerror_when_running():
    evaluator = EnsembleEvaluator(TestEnsemble(0, 2, 2, id_="0"), None)
    with pytest.raises(ValueError, match="no config for evaluator"):
        await evaluator.run_and_get_successful_realizations()


@pytest.mark.integration_test
@pytest.mark.parametrize(
    ("task, task_name"),
    [
        ("_batch_events_into_buffer", "dispatcher_task"),
        ("_process_event_buffer", "processing_task"),
    ],
)
async def test_when_task_prematurely_ends_raises_exception(
    task, task_name, make_ee_config, monkeypatch
):
    async def mock_done_prematurely(message, *args, **kwargs):
        await asyncio.sleep(0.5)

    evaluator = EnsembleEvaluator(TestEnsemble(0, 2, 2, id_="0"), make_ee_config())
    monkeypatch.setattr(
        EnsembleEvaluator,
        task,
        mock_done_prematurely,
    )
    error_msg = f"Something went wrong, {task_name} is done prematurely!"
    with pytest.raises(RuntimeError, match=error_msg):
        await evaluator.run_and_get_successful_realizations()


async def test_new_connections_are_no_problem_when_evaluator_is_closing_down(
    evaluator_to_use,
):
    (evaluator, _) = evaluator_to_use

    async def new_connection():
        await evaluator._server_done.wait()
        async with Client(evaluator._config.get_uri()):
            pass

    new_connection_task = asyncio.create_task(new_connection())
    evaluator.stop()

    await new_connection_task


@pytest.fixture
async def evaluator_to_use(make_ee_config):
    event_queue: asyncio.Queue[Event] = asyncio.Queue()
    ensemble = TestEnsemble(0, 2, 2, id_="0")
    evaluator = EnsembleEvaluator(
        ensemble, make_ee_config(use_token=False), event_handler=event_queue.put_nowait
    )
    evaluator._batching_interval = 0.5  # batching can be faster for tests
    run_task = asyncio.create_task(evaluator.run_and_get_successful_realizations())
    await evaluator._server_started
    yield (evaluator, event_queue)
    evaluator.stop()
    await run_task


@pytest.mark.integration_test
@pytest.mark.timeout(20)
async def test_restarted_jobs_do_not_have_error_msgs(evaluator_to_use):
    (evaluator, event_queue) = evaluator_to_use
    token = evaluator._config.token
    url = evaluator._config.get_uri()

    # first snapshot before any event occurs
    snapshot_event = await event_queue.get()
    assert isinstance(snapshot_event, EESnapshot)
    final_snapshot = EnsembleSnapshot.from_nested_dict(snapshot_event.snapshot)
    assert final_snapshot.status == ENSEMBLE_STATE_UNKNOWN
    # two dispatch endpoint clients connect
    async with Client(
        url,
        token=token,
        dealer_name="dispatch_from_test_1",
    ) as dispatch:
        event = ForwardModelStepRunning(
            ensemble=evaluator.ensemble.id_,
            real="0",
            fm_step="0",
            current_memory_usage=1000,
        )
        await dispatch.send(event_to_json(event))

        event = ForwardModelStepFailure(
            ensemble=evaluator.ensemble.id_,
            real="0",
            fm_step="0",
            error_msg="error",
        )
        await dispatch.send(event_to_json(event))

        while True:
            event = await event_queue.get()
            final_snapshot.update_from_event(event)
            if (
                final_snapshot.get_fm_step("0", "0")["status"]
                == FORWARD_MODEL_STATE_FAILURE
                and final_snapshot.get_fm_step("0", "0")["error"] == "error"
            ):
                break

    async with Client(
        url,
        token=token,
    ) as dispatch:
        event = ForwardModelStepSuccess(
            ensemble=evaluator.ensemble.id_,
            real="0",
            fm_step="0",
            current_memory_usage=1000,
        )
        await dispatch.send(event_to_json(event))

        while True:
            event = await event_queue.get()
            final_snapshot = final_snapshot.update_from_event(event)
            if (
                final_snapshot.status == ENSEMBLE_STATE_UNKNOWN
                and (
                    final_snapshot.get_fm_step("0", "0")["status"]
                    == FORWARD_MODEL_STATE_FINISHED
                )
                and not final_snapshot.get_fm_step("0", "0")["error"]
            ):
                break


@pytest.mark.integration_test
async def test_ensure_multi_level_events_in_order(evaluator_to_use):
    (evaluator, event_queue) = evaluator_to_use

    token = evaluator._config.token
    url = evaluator._config.get_uri()

    snapshot_event = await event_queue.get()
    assert type(snapshot_event) is EESnapshot
    async with Client(url, token=token) as dispatch:
        event = EnsembleStarted(ensemble=evaluator.ensemble.id_)
        await dispatch.send(event_to_json(event))
        event = RealizationSuccess(
            ensemble=evaluator.ensemble.id_, real="0", queue_event_type=""
        )
        await dispatch.send(event_to_json(event))
        event = RealizationSuccess(
            ensemble=evaluator.ensemble.id_, real="1", queue_event_type=""
        )
        await dispatch.send(event_to_json(event))
        event = EnsembleSucceeded(ensemble=evaluator.ensemble.id_)
        await dispatch.send(event_to_json(event))

        assert await asyncio.wait_for(evaluator._monitoring_result, timeout=5)

        # Without making too many assumptions about what events to expect, it
        # should be reasonable to expect that if an event contains information
        # about realizations, the state of the ensemble up until that point
        # should be not final (i.e. not cancelled, stopped, failed).
        ensemble_state = snapshot_event.snapshot.get("status")
        snapshot_event_received = False
        while not event_queue.empty():
            event = event_queue.get_nowait()
            assert type(event) in {EESnapshot, EESnapshotUpdate}
            # if we get an snapshot event than this need to be valid
            snapshot_event_received = True
            if "reals" in event.snapshot:
                assert ensemble_state == ENSEMBLE_STATE_STARTED
            ensemble_state = event.snapshot.get("status", ensemble_state)
        assert ensemble_state == ENSEMBLE_STATE_STOPPED
        assert snapshot_event_received is True


@given(
    num_cpu=st.integers(min_value=1, max_value=64),
    start=st.datetimes(),
    duration=st.integers(min_value=-1, max_value=1000),
    cpu_seconds=st.floats(min_value=0, max_value=2000),
)
def test_overspent_cpu_is_logged(
    num_cpu: int,
    start: datetime.datetime,
    duration: int,
    cpu_seconds: float,
):
    message = detect_overspent_cpu(
        num_cpu,
        "dummy",
        FMStepSnapshot(
            start_time=start,
            end_time=start + datetime.timedelta(seconds=duration),
            cpu_seconds=cpu_seconds,
        ),
    )
    if duration > 0 and cpu_seconds / duration > num_cpu * 1.05:
        assert "Misconfigured NUM_CPU" in message
    else:
        assert "NUM_CPU" not in message


@pytest.mark.integration_test
async def test_snapshot_on_resubmit_is_cleared(evaluator_to_use):
    (evaluator, event_queue) = evaluator_to_use
    evaluator._batching_interval = 0.4
    token = evaluator._config.token
    url = evaluator._config.get_uri()

    snapshot_event = await event_queue.get()
    assert type(snapshot_event) is EESnapshot
    async with Client(url, token=token) as dispatch:
        event = ForwardModelStepRunning(
            ensemble=evaluator.ensemble.id_,
            real="0",
            fm_step="0",
            current_memory_usage=1000,
        )
        await dispatch.send(event_to_json(event))
        event = ForwardModelStepSuccess(
            ensemble=evaluator.ensemble.id_,
            real="0",
            fm_step="0",
            current_memory_usage=1000,
        )
        await dispatch.send(event_to_json(event))
        event = ForwardModelStepRunning(
            ensemble=evaluator.ensemble.id_,
            real="0",
            fm_step="1",
            current_memory_usage=1000,
        )
        await dispatch.send(event_to_json(event))
        event = ForwardModelStepFailure(
            ensemble=evaluator.ensemble.id_,
            real="0",
            fm_step="1",
            error_msg="error",
        )
        await dispatch.send(event_to_json(event))
        event = await event_queue.get()
        snapshot = EnsembleSnapshot.from_nested_dict(event.snapshot)
        assert snapshot.get_fm_step("0", "0")["status"] == FORWARD_MODEL_STATE_FINISHED
        assert snapshot.get_fm_step("0", "1")["status"] == FORWARD_MODEL_STATE_FAILURE
        event_dict = {
            "ensemble": str(evaluator._ensemble.id_),
            "event_type": Id.REALIZATION_RESUBMIT,
            "queue_event_type": JobState.RESUBMITTING,
            "real": "0",
            "exec_hosts": "something",
        }
        await evaluator._events.put(event_from_dict(event_dict))
        event = await event_queue.get()
        snapshot = EnsembleSnapshot.from_nested_dict(event.snapshot)
        assert snapshot.get_fm_step("0", "0")["status"] == FORWARD_MODEL_STATE_INIT
        assert snapshot.get_fm_step("0", "1")["status"] == FORWARD_MODEL_STATE_INIT

    await evaluator.cancel_gracefully()
    await evaluator._monitoring_result


@pytest.mark.integration_test
async def test_signal_cancel_does_not_cause_evaluator_dispatcher_communication_to_hang(
    evaluator_to_use, monkeypatch
):
    (evaluator, event_queue) = evaluator_to_use
    evaluator._batching_interval = 0.4
    evaluator._max_batch_size = 1

    kill_all_jobs_event = asyncio.Event()
    started_cancelling_ensemble = asyncio.Event()

    async def mock_never_ending_ensemble_cancel(*args, **kwargs):
        nonlocal kill_all_jobs_event, started_cancelling_ensemble
        started_cancelling_ensemble.set()
        await kill_all_jobs_event.wait()

    monkeypatch.setattr(LegacyEnsemble, "cancel", mock_never_ending_ensemble_cancel)
    monkeypatch.setattr(Client, "DEFAULT_MAX_RETRIES", 1)
    monkeypatch.setattr(Client, "DEFAULT_ACK_TIMEOUT", 1)
    token = evaluator._config.token
    url = evaluator._config.get_uri()
    evaluator.ensemble._cancellable = True

    async with Client(url, token=token) as dispatch:
        event = ForwardModelStepRunning(
            ensemble=evaluator.ensemble.id_,
            real="0",
            fm_step="0",
            current_memory_usage=1000,
        )
        await dispatch.send(event_to_json(event))

        async def try_sending_event_from_dispatcher_while_monitor_is_cancelling_ensemble():
            await started_cancelling_ensemble.wait()
            event = ForwardModelStepSuccess(
                ensemble=evaluator.ensemble.id_,
                real="0",
                fm_step="0",
                current_memory_usage=1000,
            )
            await dispatch.send(event_to_json(event))
            kill_all_jobs_event.set()

        await asyncio.wait_for(
            asyncio.gather(
                try_sending_event_from_dispatcher_while_monitor_is_cancelling_ensemble(),
                evaluator.cancel_gracefully(),
            ),
            timeout=10,
        )

    def is_completed_snapshot(snapshot: EnsembleSnapshot) -> bool:
        try:
            assert (
                snapshot.get_fm_step("0", "0").get("status")
                == FORWARD_MODEL_STATE_FINISHED
            )
        except AssertionError:
            return False
        else:
            return True

    was_completed = False
    final_snapshot = EnsembleSnapshot()
    while True:
        event = await event_queue.get()
        final_snapshot.update_from_event(event)
        if is_completed_snapshot(final_snapshot):
            was_completed = True
            break

    assert was_completed
    assert await evaluator._monitoring_result == False
