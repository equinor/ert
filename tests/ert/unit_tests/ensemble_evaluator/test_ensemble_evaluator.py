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
    ForwardModelStepFailure,
    ForwardModelStepRunning,
    ForwardModelStepSuccess,
    RealizationResubmit,
    dispatcher_event_to_json,
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
        ("_publisher", "Publisher failed!"),
    ],
)
async def test_when_task_fails_evaluator_raises_exception(
    task, error_msg, make_ee_config, monkeypatch
):
    async def mock_failure(message, *args, **kwargs):
        raise RuntimeError(message)

    evaluator = EnsembleEvaluator(
        TestEnsemble(0, 2, 2, id_="0"),
        make_ee_config(use_token=False),
        end_event=asyncio.Event(),
    )

    monkeypatch.setattr(
        EnsembleEvaluator,
        task,
        partial(mock_failure, error_msg),
    )
    with pytest.raises(RuntimeError, match=error_msg):
        await evaluator.run_and_get_successful_realizations()


async def test_evaluator_raises_on_invalid_dispatch_event(make_ee_config):
    evaluator = EnsembleEvaluator(
        TestEnsemble(0, 2, 2, id_="0"),
        make_ee_config(),
        end_event=asyncio.Event(),
    )

    with pytest.raises(ValidationError):
        await evaluator.handle_dispatch(b"dispatcher-1", b"This is not an event!!")


async def test_evaluator_handles_dispatchers_connected(
    make_ee_config,
):
    evaluator = EnsembleEvaluator(
        TestEnsemble(0, 2, 2, id_="0"),
        make_ee_config(),
        end_event=asyncio.Event(),
    )

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

        evaluator = EnsembleEvaluator(
            TestEnsemble(0, 2, 2, id_="0"),
            ee_config,
            end_event=asyncio.Event(),
        )
        with pytest.raises(
            zmq.error.ZMQBindError, match="Could not bind socket to random port"
        ):
            await evaluator.run_and_get_successful_realizations()
    finally:
        socket.close()
        ctx.destroy(linger=0)


async def test_no_config_raises_valueerror_when_running():
    evaluator = EnsembleEvaluator(
        TestEnsemble(0, 2, 2, id_="0"),
        None,
        end_event=asyncio.Event(),
    )
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

    event_queue = asyncio.Queue()

    def event_handler(snapshot):
        event_queue.put_nowait(snapshot)

    evaluator = EnsembleEvaluator(
        TestEnsemble(0, 2, 2, id_="0"),
        make_ee_config(),
        end_event=asyncio.Event(),
        event_handler=event_handler,
    )
    monkeypatch.setattr(
        EnsembleEvaluator,
        task,
        mock_done_prematurely,
    )
    error_msg = f"Something went wrong, {task_name} is done prematurely!"
    with pytest.raises(RuntimeError, match=error_msg):
        await evaluator.run_and_get_successful_realizations()


@pytest.mark.integration_test
@pytest.mark.timeout(20)
async def test_restarted_jobs_do_not_have_error_msgs(evaluator_to_use):
    event_queue: asyncio.Queue[EESnapshot | EESnapshotUpdate] = asyncio.Queue()
    async with evaluator_to_use(event_handler=event_queue.put_nowait) as evaluator:
        token = evaluator._config.token
        url = evaluator._config.get_uri()

        snapshot_event = await event_queue.get()
        snapshot = EnsembleSnapshot.from_nested_dict(snapshot_event.snapshot)
        assert snapshot.status == ENSEMBLE_STATE_UNKNOWN
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
            await dispatch.send(dispatcher_event_to_json(event))

            event = ForwardModelStepFailure(
                ensemble=evaluator.ensemble.id_,
                real="0",
                fm_step="0",
                error_msg="error",
            )
            await dispatch.send(dispatcher_event_to_json(event))

        def is_completed_snapshot(snapshot: EnsembleSnapshot) -> bool:
            try:
                assert (
                    snapshot.get_fm_step("0", "0")["status"]
                    == FORWARD_MODEL_STATE_FAILURE
                )
                assert snapshot.get_fm_step("0", "0")["error"] == "error"
            except AssertionError:
                return False
            else:
                return True

        while True:
            event = await event_queue.get()
            snapshot.update_from_event(event)
            if is_completed_snapshot(snapshot):
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
            await dispatch.send(dispatcher_event_to_json(event))

            def check_if_final_snapshot_is_complete(snapshot: EnsembleSnapshot) -> bool:
                try:
                    assert snapshot.status == ENSEMBLE_STATE_UNKNOWN
                    assert (
                        snapshot.get_fm_step("0", "0")["status"]
                        == FORWARD_MODEL_STATE_FINISHED
                    )
                    assert not snapshot.get_fm_step("0", "0")["error"]
                except AssertionError:
                    return False
                else:
                    return True

            while True:
                event = await event_queue.get()
                snapshot = snapshot.update_from_event(event)
                if check_if_final_snapshot_is_complete(snapshot):
                    break


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
    if duration > 30 and cpu_seconds / duration > num_cpu * 1.05:
        assert "Misconfigured NUM_CPU" in message
    else:
        assert "NUM_CPU" not in message


@pytest.mark.integration_test
async def test_snapshot_on_resubmit_is_cleared(evaluator_to_use):
    event_queue: asyncio.Queue[EESnapshot | EESnapshotUpdate] = asyncio.Queue()
    async with evaluator_to_use(event_handler=event_queue.put_nowait) as evaluator:
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
            await dispatch.send(dispatcher_event_to_json(event))
            event = ForwardModelStepSuccess(
                ensemble=evaluator.ensemble.id_,
                real="0",
                fm_step="0",
                current_memory_usage=1000,
            )
            await dispatch.send(dispatcher_event_to_json(event))
            event = ForwardModelStepRunning(
                ensemble=evaluator.ensemble.id_,
                real="0",
                fm_step="1",
                current_memory_usage=1000,
            )
            await dispatch.send(dispatcher_event_to_json(event))
            event = ForwardModelStepFailure(
                ensemble=evaluator.ensemble.id_,
                real="0",
                fm_step="1",
                error_msg="error",
            )
            await dispatch.send(dispatcher_event_to_json(event))
            event = await event_queue.get()
            snapshot = EnsembleSnapshot.from_nested_dict(event.snapshot)
            assert (
                snapshot.get_fm_step("0", "0")["status"] == FORWARD_MODEL_STATE_FINISHED
            )
            assert (
                snapshot.get_fm_step("0", "1")["status"] == FORWARD_MODEL_STATE_FAILURE
            )
            await evaluator._events.put(
                RealizationResubmit(
                    ensemble=evaluator.ensemble.id_,
                    queue_event_type=JobState.RESUBMITTING,
                    real="0",
                    exec_hosts="something",
                )
            )
            event = await event_queue.get()
            snapshot = EnsembleSnapshot.from_nested_dict(event.snapshot)
            assert snapshot.get_fm_step("0", "0")["status"] == FORWARD_MODEL_STATE_INIT
            assert snapshot.get_fm_step("0", "1")["status"] == FORWARD_MODEL_STATE_INIT


@pytest.mark.integration_test
async def test_signal_cancel_does_not_cause_evaluator_dispatcher_communication_to_hang(
    evaluator_to_use, monkeypatch
):
    event_queue: asyncio.Queue[EESnapshot | EESnapshotUpdate] = asyncio.Queue()
    async with evaluator_to_use(event_handler=event_queue.put_nowait) as evaluator:
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
            await dispatch.send(dispatcher_event_to_json(event))

            async def try_sending_event_from_dispatcher_while_monitor_is_cancelling_ensemble():  # noqa: E501
                await started_cancelling_ensemble.wait()
                event = ForwardModelStepSuccess(
                    ensemble=evaluator.ensemble.id_,
                    real="0",
                    fm_step="0",
                    current_memory_usage=1000,
                )
                await dispatch.send(dispatcher_event_to_json(event))
                kill_all_jobs_event.set()

            await asyncio.wait_for(
                asyncio.gather(
                    try_sending_event_from_dispatcher_while_monitor_is_cancelling_ensemble(),
                    evaluator._signal_cancel(),
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
