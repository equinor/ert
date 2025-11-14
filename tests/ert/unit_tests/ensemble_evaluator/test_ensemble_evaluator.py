import asyncio
import datetime
import logging
import uuid
from functools import partial
from threading import Event
from unittest.mock import MagicMock

import pytest
import zmq.asyncio
from hypothesis import given
from hypothesis import strategies as st
from pydantic import ValidationError
from pytest import MonkeyPatch

from _ert.events import (
    EESnapshot,
    EESnapshotUpdate,
    ForwardModelStepFailure,
    ForwardModelStepRunning,
    ForwardModelStepStart,
    ForwardModelStepSuccess,
    RealizationFailed,
    RealizationResubmit,
    RealizationSuccess,
    dispatcher_event_to_json,
)
from _ert.forward_model_runner.client import (
    CONNECT_MSG,
    DISCONNECT_MSG,
    Client,
)
from ert.config.ert_config import ErtConfig
from ert.config.queue_config import QueueConfig
from ert.ensemble_evaluator import (
    EnsembleEvaluator,
    EnsembleSnapshot,
    FMStepSnapshot,
    identifiers,
    state,
)
from ert.ensemble_evaluator._ensemble import LegacyEnsemble
from ert.ensemble_evaluator.config import EvaluatorServerConfig
from ert.ensemble_evaluator.evaluator import UserCancelled, detect_overspent_cpu
from ert.ensemble_evaluator.state import (
    ENSEMBLE_STATE_CANCELLED,
    ENSEMBLE_STATE_UNKNOWN,
    FORWARD_MODEL_STATE_FAILURE,
    FORWARD_MODEL_STATE_FINISHED,
    FORWARD_MODEL_STATE_INIT,
)
from ert.scheduler import JobState
from ert.scheduler.job import Job
from ert.scheduler.scheduler import Scheduler

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
        end_event=Event(),
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
        end_event=Event(),
    )

    with pytest.raises(ValidationError):
        await evaluator.handle_dispatch(b"dispatcher-1", b"This is not an event!!")


async def test_evaluator_handles_dispatchers_connected(
    make_ee_config,
):
    evaluator = EnsembleEvaluator(
        TestEnsemble(0, 2, 2, id_="0"),
        make_ee_config(),
        end_event=Event(),
    )

    await evaluator.handle_dispatch(b"dispatcher-iens-1", CONNECT_MSG)
    await evaluator.handle_dispatch(b"dispatcher-iens-2", CONNECT_MSG)
    assert not evaluator._dispatchers_empty.is_set()
    assert evaluator._dispatchers_connected == {
        b"dispatcher-iens-1",
        b"dispatcher-iens-2",
    }
    await evaluator.handle_dispatch(b"dispatcher-iens-1", DISCONNECT_MSG)
    await evaluator.handle_dispatch(b"dispatcher-iens-2", DISCONNECT_MSG)
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
            end_event=Event(),
        )
        with pytest.raises(
            zmq.error.ZMQBindError, match="Could not bind socket to random port"
        ):
            await evaluator.run_and_get_successful_realizations()
    finally:
        socket.close()
        ctx.destroy(linger=0)


async def test_no_config_raises_valueerror():
    with pytest.raises(ValueError, match="no config for evaluator"):
        EnsembleEvaluator(
            TestEnsemble(0, 2, 2, id_="0"),
            None,
            end_event=Event(),
        )


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
        end_event=Event(),
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


@pytest.fixture(name="evaluator_to_use")
async def evaluator_to_use_fixture(monkeypatch, make_ee_config):
    ensemble = TestEnsemble(0, 2, 2, id_="0")

    async def empty_mock_function(*args, **kwargs):
        pass

    monkeypatch.setattr(EnsembleEvaluator, "evaluate", empty_mock_function)
    monkeypatch.setattr(EnsembleEvaluator, "DEFAULT_SLEEP_PERIOD", 0.05)
    monkeypatch.setattr(Scheduler, "BATCH_KILLING_INTERVAL", 0)
    event_queue = asyncio.Queue()
    evaluator = EnsembleEvaluator(
        ensemble, make_ee_config(use_token=False), Event(), event_queue.put_nowait
    )

    evaluator._scheduler.kill_all_jobs = empty_mock_function
    evaluator._scheduler._running.set()
    evaluator._batching_interval = 0.05  # batching can be faster for tests
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

    snapshot_event = await event_queue.get()
    snapshot = EnsembleSnapshot.from_nested_dict(snapshot_event.snapshot)
    assert snapshot.status == ENSEMBLE_STATE_UNKNOWN
    # two dispatch endpoint clients connect
    async with Client(
        url,
        token=token,
        dealer_name=f"dispatch-real-0-{uuid.uuid4().hex[:6]}",
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
                snapshot.get_fm_step("0", "0").get("status")
                == FORWARD_MODEL_STATE_FAILURE
            )
            assert snapshot.get_fm_step("0", "0").get("error") == "error"
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
        url, token=token, dealer_name=f"dispatch-real-0-{uuid.uuid4().hex[:6]}"
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
                    snapshot.get_fm_step("0", "0").get("status")
                    == FORWARD_MODEL_STATE_FINISHED
                )
                assert not snapshot.get_fm_step("0", "0").get("error")
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
    (evaluator, event_queue) = evaluator_to_use
    token = evaluator._config.token
    url = evaluator._config.get_uri()

    event = await event_queue.get()
    assert type(event) is EESnapshot
    main_snapshot = EnsembleSnapshot.from_nested_dict(event.snapshot)
    async with Client(
        url, token=token, dealer_name=f"dispatch-real-0-{uuid.uuid4().hex[:6]}"
    ) as dispatch:
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
        main_snapshot.update_from_event(event)
        # main_snapshot = EnsembleSnapshot.from_nested_dict(event.snapshot)
        while not event_queue.empty():
            event = await event_queue.get()
            main_snapshot.update_from_event(event)

        assert (
            main_snapshot.get_fm_step("0", "0").get("status")
            == FORWARD_MODEL_STATE_FINISHED
        )
        assert (
            main_snapshot.get_fm_step("0", "1").get("status")
            == FORWARD_MODEL_STATE_FAILURE
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
        main_snapshot.update_from_event(event)
        assert (
            main_snapshot.get_fm_step("0", "0").get("status")
            == FORWARD_MODEL_STATE_INIT
        )
        assert (
            main_snapshot.get_fm_step("0", "1").get("status")
            == FORWARD_MODEL_STATE_INIT
        )


@pytest.mark.integration_test
async def test_signal_cancel_does_not_cause_evaluator_dispatcher_communication_to_hang(
    evaluator_to_use, monkeypatch
):
    (evaluator, event_queue) = evaluator_to_use
    evaluator._batching_interval = 0.4
    evaluator._max_batch_size = 1

    terminated_all_dispatchers_event = asyncio.Event()
    started_terminating_all_dispatchers = asyncio.Event()

    async def mock_never_ending_termination_of_dispatchers(*args, **kwargs):
        nonlocal terminated_all_dispatchers_event, started_terminating_all_dispatchers
        started_terminating_all_dispatchers.set()
        await terminated_all_dispatchers_event.wait()

    monkeypatch.setattr(
        EnsembleEvaluator,
        "_terminate_all_dispatchers",
        mock_never_ending_termination_of_dispatchers,
    )
    monkeypatch.setattr(Client, "DEFAULT_MAX_RETRIES", 1)
    monkeypatch.setattr(Client, "DEFAULT_ACK_TIMEOUT", 1)
    token = evaluator._config.token
    url = evaluator._config.get_uri()
    evaluator.ensemble._cancellable = True
    async with Client(
        url, token=token, dealer_name=f"dispatch-real-0-{uuid.uuid4().hex[:6]}"
    ) as dispatch:
        event = ForwardModelStepRunning(
            ensemble=evaluator.ensemble.id_,
            real="0",
            fm_step="0",
            current_memory_usage=1000,
        )
        await dispatch.send(dispatcher_event_to_json(event))

        async def try_sending_event_from_dispatcher_while_evaluator_is_terminating_dispatchers():  # noqa: E501
            await started_terminating_all_dispatchers.wait()
            event = ForwardModelStepSuccess(
                ensemble=evaluator.ensemble.id_,
                real="0",
                fm_step="0",
                current_memory_usage=1000,
            )
            await dispatch.send(dispatcher_event_to_json(event))
            terminated_all_dispatchers_event.set()

        await asyncio.wait_for(
            asyncio.gather(
                try_sending_event_from_dispatcher_while_evaluator_is_terminating_dispatchers(),
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


@pytest.mark.timeout(15)
async def test_signal_cancel_sends_terminate_message_to_dispatchers(evaluator_to_use):
    (evaluator, _) = evaluator_to_use
    token = evaluator._config.token
    url = evaluator._config.get_uri()
    evaluator.ensemble._cancellable = True
    async with (
        Client(
            url, token=token, dealer_name=f"dispatch-real-0-{uuid.uuid4().hex[:6]}"
        ) as dispatcher_0,
        Client(
            url, token=token, dealer_name=f"dispatch-real-1-{uuid.uuid4().hex[:6]}"
        ) as dispatcher_1,
    ):
        await evaluator._signal_cancel()
        assert await asyncio.wait_for(
            dispatcher_0.received_terminate_message.wait(), timeout=5
        )
        assert await asyncio.wait_for(
            dispatcher_1.received_terminate_message.wait(), timeout=5
        )


@pytest.mark.timeout(10)
@pytest.mark.integration_test
async def test_signal_cancel_terminates_fm_dispatcher_with_terminate_message(
    tmpdir, monkeypatch, make_ensemble, caplog
):
    caplog.set_level(logging.INFO)
    driver_kill_was_called = asyncio.Event()

    async def mock_driver_kill(*args, **kwargs):
        nonlocal driver_kill_was_called
        driver_kill_was_called.set()

    num_reals = 1
    num_jobs = 1
    sleep_period = 60
    ensemble: LegacyEnsemble = make_ensemble(
        monkeypatch, tmpdir, num_reals, num_jobs, sleep_period
    )
    config = EvaluatorServerConfig(use_token=False)
    event_queue: asyncio.Queue[EESnapshot | EESnapshotUpdate] = asyncio.Queue()
    monkeypatch.setattr(EnsembleEvaluator, "DEFAULT_SLEEP_PERIOD", 0.05)
    evaluator = EnsembleEvaluator(ensemble, config, Event(), event_queue.put_nowait)
    evaluator._batching_interval = 0.05
    run_task = asyncio.create_task(evaluator.run_and_get_successful_realizations())
    await evaluator._server_started

    async def dispatcher_is_running():
        nonlocal evaluator
        while True:
            if not evaluator._dispatchers_empty.is_set():
                break
            await asyncio.sleep(0.1)

    await asyncio.wait_for(dispatcher_is_running(), timeout=5)

    async def cancel_evaluator_after_getting_initial_event_and_wait_for_confirmation():
        nonlocal event_queue
        while True:
            event = await event_queue.get()
            if type(event) in {
                EESnapshotUpdate,
                EESnapshot,
            }:
                if (
                    event.snapshot.get(identifiers.STATUS)
                    == state.ENSEMBLE_STATE_STARTED
                ):
                    evaluator._scheduler.driver.kill = mock_driver_kill
                    evaluator._end_event.set()
                elif event.snapshot.get(identifiers.STATUS) == ENSEMBLE_STATE_CANCELLED:
                    break

    await asyncio.wait_for(
        cancel_evaluator_after_getting_initial_event_and_wait_for_confirmation(),
        timeout=8,
    )

    await run_task
    assert len(evaluator._ensemble.get_successful_realizations()) == 0
    assert not driver_kill_was_called.is_set()
    assert evaluator._ensemble.status == ENSEMBLE_STATE_CANCELLED
    assert isinstance(evaluator._evaluation_result.exception(), UserCancelled)

    async def wait_for_log_message():
        nonlocal caplog
        while "Realization 0 was killed by the evaluator" not in caplog.text:  # noqa: ASYNC110
            await asyncio.sleep(0.1)

    await asyncio.wait_for(wait_for_log_message(), timeout=10)
    # assert "Realization 0 was killed by the evaluator" in caplog.text


@pytest.mark.timeout(10)
@pytest.mark.integration_test
async def test_signal_cancel_terminates_fm_dispatcher_with_scheduler_as_fallback(
    tmpdir, monkeypatch: MonkeyPatch, make_ensemble, caplog
):
    caplog.set_level(logging.INFO)
    send_terminate_was_called = asyncio.Event()

    async def empty_send_terminate_messages(*args, **kwargs):
        nonlocal send_terminate_was_called
        send_terminate_was_called.set()

    num_reals = 1
    num_jobs = 1
    sleep_period = 60
    ensemble: LegacyEnsemble = make_ensemble(
        monkeypatch, tmpdir, num_reals, num_jobs, sleep_period
    )
    config = EvaluatorServerConfig(use_token=False)
    event_queue: asyncio.Queue[EESnapshot | EESnapshotUpdate] = asyncio.Queue()
    monkeypatch.setattr(Job, "WAIT_PERIOD_FOR_TERM_MESSAGE_TO_CANCEL", 0)
    monkeypatch.setattr(Scheduler, "BATCH_KILLING_INTERVAL", 0)
    monkeypatch.setattr(
        EnsembleEvaluator,
        "_send_terminate_message_to_dispatchers",
        empty_send_terminate_messages,
    )
    monkeypatch.setattr(EnsembleEvaluator, "DEFAULT_SLEEP_PERIOD", 0.05)
    evaluator = EnsembleEvaluator(ensemble, config, Event(), event_queue.put_nowait)
    evaluator._batching_interval = 0.05
    run_task = asyncio.create_task(evaluator.run_and_get_successful_realizations())
    await evaluator._server_started

    async def cancel_evaluator_after_getting_initial_event_and_wait_for_confirmation():
        nonlocal event_queue
        while True:
            event = await event_queue.get()
            if type(event) in {
                EESnapshotUpdate,
                EESnapshot,
            }:
                if (
                    event.snapshot.get(identifiers.STATUS)
                    == state.ENSEMBLE_STATE_STARTED
                ):
                    evaluator._end_event.set()
                elif event.snapshot.get(identifiers.STATUS) == ENSEMBLE_STATE_CANCELLED:
                    break

    await asyncio.wait_for(
        cancel_evaluator_after_getting_initial_event_and_wait_for_confirmation(),
        timeout=5,
    )
    await run_task

    assert send_terminate_was_called.is_set()
    assert evaluator._ensemble.status == ENSEMBLE_STATE_CANCELLED
    assert isinstance(evaluator._evaluation_result.exception(), UserCancelled)
    assert (
        "Realization 0 was not killed gracefully by TERM message. "
        "Killing it with the driver"
    ) in caplog.text


async def test_queue_config_properties_propagated_to_scheduler_from_ensemble(
    tmpdir, make_ensemble, monkeypatch
):
    num_reals = 1
    mocked_scheduler = MagicMock(return_value=None)
    mocked_scheduler.__class__ = Scheduler
    monkeypatch.setattr(Scheduler, "__init__", mocked_scheduler)
    ensemble = make_ensemble(monkeypatch, tmpdir, num_reals, 2)
    ensemble._config = MagicMock()
    ensemble._scheduler = mocked_scheduler

    # The properties we want to propagate from QueueConfig to the Scheduler object:
    monkeypatch.setattr(QueueConfig, "submit_sleep", 33)
    monkeypatch.setattr(QueueConfig, "max_running", 44)
    ensemble._queue_config.max_submit = 55

    # The function under test:
    _ = EnsembleEvaluator(ensemble, MagicMock(), Event())

    # Assert properties successfully propagated:
    assert Scheduler.__init__.call_args.kwargs["submit_sleep"] == 33
    assert Scheduler.__init__.call_args.kwargs["max_running"] == 44
    assert Scheduler.__init__.call_args.kwargs["max_submit"] == 55


async def test_log_forward_model_steps_with_missing_status_updates(
    monkeypatch: MonkeyPatch, tmpdir, caplog, make_ensemble
):
    mocked_config = MagicMock(spec=ErtConfig)
    num_reals = 11
    num_fm_steps = 3
    ensemble: LegacyEnsemble = make_ensemble(
        monkeypatch, tmpdir, num_reals, num_fm_steps
    )
    router_port = 111111
    mocked_config.router_port = router_port
    working_machine_name = "working_cluster_machine"
    evaluator_host = "foo_evaluator_host"
    evaluator = EnsembleEvaluator(ensemble, mocked_config, Event())
    monkeypatch.setattr(
        "ert.ensemble_evaluator.evaluator.get_machine_name",
        lambda *args: evaluator_host,
    )
    # One realization passed with no connection problems and reports successfully
    fm_events = []
    for fm_step_id in range(num_fm_steps):
        fm_events.extend(
            [
                ForwardModelStepStart(
                    ensemble=ensemble.id_, real="10", fm_step=str(fm_step_id)
                ),
                ForwardModelStepRunning(
                    ensemble=ensemble.id_, real="10", fm_step=str(fm_step_id)
                ),
                ForwardModelStepSuccess(
                    ensemble=ensemble.id_, real="10", fm_step=str(fm_step_id)
                ),
            ]
        )
    evaluator._ensemble.update_snapshot(fm_events)
    driver_events = [
        RealizationSuccess(
            real="10", ensemble=ensemble.id_, exec_hosts=working_machine_name
        )
    ]
    for i in range(10):
        driver_events.append(
            RealizationFailed(
                real=str(i),
                ensemble=ensemble.id_,
                exec_hosts="foo_cluster_machine"
                if i % 2 == 0
                else "bar_cluster_machine",
            )
        )

    evaluator._ensemble.update_snapshot(driver_events)
    evaluator._log_forward_model_steps_with_missing_status_updates()

    assert "working_cluster_machine" not in caplog.text
    assert "'10'" not in caplog.text  # The realization that passed with no problems
    assert (
        "Ensemble finished, but there were missing ForwardModelStep status updates "
        "for some realization(s) from some host(s) ({'foo_cluster_machine': ['0', "
        "'2', '4', '6', '8'], 'bar_cluster_machine': ['1', '3', '5', '7', '9']}). "
        f"There could be connectivity issues to evaluator running on port "
        f"{router_port} on host {evaluator_host}"
    ) in caplog.text
