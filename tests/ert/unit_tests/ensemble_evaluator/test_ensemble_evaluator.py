import asyncio
import datetime
from functools import partial
from typing import cast
from unittest.mock import MagicMock, patch

import pytest
import zmq.asyncio
from hypothesis import given
from hypothesis import strategies as st
from pydantic import ValidationError

from _ert.events import (
    EESnapshot,
    EESnapshotUpdate,
    EETerminated,
    EnsembleStarted,
    EnsembleSucceeded,
    ForwardModelStepFailure,
    ForwardModelStepRunning,
    ForwardModelStepStart,
    ForwardModelStepSuccess,
    RealizationFailed,
    RealizationResubmit,
    RealizationSuccess,
    event_to_json,
)
from _ert.forward_model_runner.client import (
    ACK_MSG,
    CONNECT_MSG,
    DISCONNECT_MSG,
    HEARTBEAT_MSG,
    Client,
)
from ert.config.ert_config import ErtConfig
from ert.ensemble_evaluator import (
    EnsembleEvaluator,
    EnsembleSnapshot,
    FMStepSnapshot,
    Monitor,
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
    FORWARD_MODEL_STATE_RUNNING,
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
        TestEnsemble(0, 2, 2, id_="0"), make_ee_config(use_token=False)
    )

    monkeypatch.setattr(
        EnsembleEvaluator,
        task,
        partial(mock_failure, error_msg),
    )
    with pytest.raises(RuntimeError, match=error_msg):
        await evaluator.run_and_get_successful_realizations()


async def test_evaluator_raises_on_invalid_dispatch_event(make_ee_config):
    evaluator = EnsembleEvaluator(TestEnsemble(0, 2, 2, id_="0"), make_ee_config())

    with pytest.raises(ValidationError):
        await evaluator.handle_dispatch(b"dispatcher-1", b"This is not an event!!")


async def test_evaluator_handles_dispatchers_connected(
    make_ee_config,
):
    evaluator = EnsembleEvaluator(TestEnsemble(0, 2, 2, id_="0"), make_ee_config())

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
        ("_publisher", "publisher_task"),
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
    evaluator = evaluator_to_use

    async def new_connection():
        await evaluator._server_done.wait()
        async with Monitor(evaluator._config.get_uri()):
            pass

    new_connection_task = asyncio.create_task(new_connection())
    evaluator.stop()

    await new_connection_task


@pytest.fixture(name="evaluator_to_use")
async def evaluator_to_use_fixture(make_ee_config):
    ensemble = TestEnsemble(0, 2, 2, id_="0")
    evaluator = EnsembleEvaluator(ensemble, make_ee_config(use_token=False))
    evaluator._batching_interval = 0.5  # batching can be faster for tests
    run_task = asyncio.create_task(evaluator.run_and_get_successful_realizations())
    await evaluator._server_started
    yield evaluator
    evaluator.stop()
    await run_task


@pytest.mark.integration_test
@pytest.mark.timeout(20)
async def test_restarted_jobs_do_not_have_error_msgs(evaluator_to_use):
    evaluator = evaluator_to_use
    token = evaluator._config.token
    url = evaluator._config.get_uri()

    async with Monitor(url, token) as monitor:
        # first snapshot before any event occurs
        events = monitor.track()
        snapshot_event = await anext(events)
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
            await dispatch.send(event_to_json(event))

            event = ForwardModelStepFailure(
                ensemble=evaluator.ensemble.id_,
                real="0",
                fm_step="0",
                error_msg="error",
            )
            await dispatch.send(event_to_json(event))

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

        final_snapshot = EnsembleSnapshot()
        async for event in monitor.track():
            final_snapshot.update_from_event(event)
            if is_completed_snapshot(final_snapshot):
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

    # reconnect new monitor
    async with Monitor(url, token) as new_monitor:

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

        final_snapshot = EnsembleSnapshot()
        async for event in new_monitor.track():
            final_snapshot = final_snapshot.update_from_event(event)
            if check_if_final_snapshot_is_complete(final_snapshot):
                break


@pytest.mark.timeout(20)
@pytest.mark.integration_test
async def test_new_monitor_can_pick_up_where_we_left_off(evaluator_to_use):
    evaluator = evaluator_to_use
    token = evaluator._config.token
    url = evaluator._config.get_uri()

    async with Monitor(url, token) as monitor:
        async with (
            Client(
                url,
                token=token,
            ) as dispatch1,
            Client(
                url,
                token=token,
            ) as dispatch2,
        ):
            # first dispatch endpoint client informs that forward model 0 is running
            event = ForwardModelStepRunning(
                ensemble=evaluator.ensemble.id_,
                real="0",
                fm_step="0",
                current_memory_usage=1000,
            )
            await dispatch1.send(event_to_json(event))
            # second dispatch endpoint client informs that forward model 0 is running
            event = ForwardModelStepRunning(
                ensemble=evaluator.ensemble.id_,
                real="1",
                fm_step="0",
                current_memory_usage=1000,
            )
            await dispatch2.send(event_to_json(event))
            # second dispatch endpoint client informs that forward model 1 is running
            event = ForwardModelStepRunning(
                ensemble=evaluator.ensemble.id_,
                real="1",
                fm_step="1",
                current_memory_usage=1000,
            )
            await dispatch2.send(event_to_json(event))

        final_snapshot = EnsembleSnapshot()

        def check_if_all_fm_running(snapshot: EnsembleSnapshot) -> bool:
            try:
                assert (
                    snapshot.get_fm_step("0", "0")["status"]
                    == FORWARD_MODEL_STATE_RUNNING
                )
                assert (
                    snapshot.get_fm_step("1", "0")["status"]
                    == FORWARD_MODEL_STATE_RUNNING
                )
                assert (
                    snapshot.get_fm_step("1", "1")["status"]
                    == FORWARD_MODEL_STATE_RUNNING
                )
            except AssertionError:
                return False
            else:
                return True

        async for event in monitor.track():
            final_snapshot = final_snapshot.update_from_event(event)
            if check_if_all_fm_running(final_snapshot):
                break
        assert final_snapshot.status == ENSEMBLE_STATE_UNKNOWN

        # take down first monitor by leaving context

    async with Client(
        url,
        token=token,
    ) as dispatch2:
        # second dispatch endpoint client informs that job 0 is done
        event = ForwardModelStepSuccess(
            ensemble=evaluator.ensemble.id_,
            real="1",
            fm_step="0",
            current_memory_usage=1000,
        )
        await dispatch2.send(event_to_json(event))
        # second dispatch endpoint client informs that job 1 is failed
        event = ForwardModelStepFailure(
            ensemble=evaluator.ensemble.id_, real="1", fm_step="1", error_msg="error"
        )
        await dispatch2.send(event_to_json(event))

    def check_if_final_snapshot_is_complete(final_snapshot: EnsembleSnapshot) -> bool:
        try:
            assert final_snapshot.status == ENSEMBLE_STATE_UNKNOWN
            assert (
                final_snapshot.get_fm_step("0", "0")["status"]
                == FORWARD_MODEL_STATE_RUNNING
            )
            assert (
                final_snapshot.get_fm_step("1", "0")["status"]
                == FORWARD_MODEL_STATE_FINISHED
            )
            assert (
                final_snapshot.get_fm_step("1", "1")["status"]
                == FORWARD_MODEL_STATE_FAILURE
            )
        except AssertionError:
            return False
        else:
            return True

    # reconnect new monitor
    async with Monitor(url, token) as new_monitor:
        final_snapshot = EnsembleSnapshot()
        async for event in new_monitor.track():
            final_snapshot = final_snapshot.update_from_event(event)
            if check_if_final_snapshot_is_complete(final_snapshot):
                break


@patch("ert.ensemble_evaluator.evaluator.HEARTBEAT_TIMEOUT", 0.1)
@pytest.mark.integration_test
async def test_monitor_receive_heartbeats(evaluator_to_use):
    evaluator = evaluator_to_use
    token = evaluator._config.token
    url = evaluator._config.get_uri()
    received_heartbeats = 0

    async def mock_receiver(self):
        nonlocal received_heartbeats
        while True:
            _, raw_msg = await self.socket.recv_multipart()
            if raw_msg == ACK_MSG:
                self._ack_event.set()
            elif raw_msg == HEARTBEAT_MSG:
                received_heartbeats += 1

    with patch.object(Monitor, "_receiver", mock_receiver):
        async with Monitor(url, token) as monitor:
            await asyncio.sleep(0.5)
            await monitor.signal_done()
    assert received_heartbeats > 1, (
        "we should have received at least 2 heartbeats in 0.5 secs!"
    )


async def test_dispatch_endpoint_clients_can_connect_and_monitor_can_shut_down_evaluator(  # noqa: E501
    evaluator_to_use,
):
    evaluator = evaluator_to_use
    evaluator._batching_interval = 0.1

    evaluator._max_batch_size = 4
    token = evaluator._config.token
    url = evaluator._config.get_uri()
    async with Monitor(url, token) as monitor:
        events = monitor.track()

        # first snapshot before any event occurs
        snapshot_event = await anext(events)
        assert type(snapshot_event) is EESnapshot
        snapshot = EnsembleSnapshot.from_nested_dict(snapshot_event.snapshot)
        assert snapshot.status == ENSEMBLE_STATE_UNKNOWN
        # two dispatch endpoint clients connect
        async with (
            Client(
                url,
                token=token,
            ) as dispatch1,
            Client(
                url,
                token=token,
            ) as dispatch2,
        ):
            # first dispatch endpoint client informs that real 0 fm 0 is running
            event = ForwardModelStepRunning(
                ensemble=evaluator.ensemble.id_,
                real="0",
                fm_step="0",
                current_memory_usage=1000,
            )
            await dispatch1.send(event_to_json(event))
            # second dispatch endpoint client informs that real 1 fm 0 is running
            event = ForwardModelStepRunning(
                ensemble=evaluator.ensemble.id_,
                real="1",
                fm_step="0",
                current_memory_usage=1000,
            )
            await dispatch2.send(event_to_json(event))
            # second dispatch endpoint client informs that real 1 fm 0 is done
            event = ForwardModelStepSuccess(
                ensemble=evaluator.ensemble.id_,
                real="1",
                fm_step="0",
                current_memory_usage=1000,
            )
            await dispatch2.send(event_to_json(event))
            # second dispatch endpoint client informs that real 1 fm 1 is failed
            event = ForwardModelStepFailure(
                ensemble=evaluator.ensemble.id_,
                real="1",
                fm_step="1",
                error_msg="error",
            )
            await dispatch2.send(event_to_json(event))

            snapshot = EnsembleSnapshot.from_nested_dict({})

            def is_completed_snapshot(snapshot):
                try:
                    assert (
                        snapshot.get_fm_step("1", "0").get("status")
                        == FORWARD_MODEL_STATE_FINISHED
                    )
                    assert (
                        snapshot.get_fm_step("0", "0").get("status")
                        == FORWARD_MODEL_STATE_RUNNING
                    )
                    assert (
                        snapshot.get_fm_step("1", "1").get("status")
                        == FORWARD_MODEL_STATE_FAILURE
                    )
                except AssertionError:
                    return False
                else:
                    return True

            async for event in events:
                snapshot.update_from_event(event)
                if is_completed_snapshot(snapshot):
                    break
        # a second monitor connects
        async with Monitor(url, token) as monitor2:
            events2 = monitor2.track()
            full_snapshot_event = await anext(events2)
            event = cast(EESnapshot, full_snapshot_event)
            snapshot = EnsembleSnapshot.from_nested_dict(event.snapshot)
            assert snapshot.status == ENSEMBLE_STATE_UNKNOWN
            assert (
                snapshot.get_fm_step("1", "0")["status"] == FORWARD_MODEL_STATE_FINISHED
            )
            assert (
                snapshot.get_fm_step("0", "0")["status"] == FORWARD_MODEL_STATE_RUNNING
            )
            assert (
                snapshot.get_fm_step("1", "1")["status"] == FORWARD_MODEL_STATE_FAILURE
            )

            # one monitor requests that server exit
            await monitor.signal_cancel()

            # both monitors should get a terminated event
            terminated = await anext(events)
            terminated2 = await anext(events2)
            assert type(terminated) is EETerminated
            assert type(terminated2) is EETerminated

            if not monitor._event_queue.empty():
                event = await monitor._event_queue.get()
                raise AssertionError(f"got unexpected event {event} from monitor")

            if not monitor2._event_queue.empty():
                event = await monitor2._event_queue.get()
                raise AssertionError(f"got unexpected event {event} from monitor2")


@pytest.mark.integration_test
async def test_ensure_multi_level_events_in_order(evaluator_to_use):
    evaluator = evaluator_to_use

    token = evaluator._config.token
    url = evaluator._config.get_uri()

    async with Monitor(url, token) as monitor:
        events = monitor.track()

        snapshot_event = await anext(events)
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

        await monitor.signal_done()

        # Without making too many assumptions about what events to expect, it
        # should be reasonable to expect that if an event contains information
        # about realizations, the state of the ensemble up until that point
        # should be not final (i.e. not cancelled, stopped, failed).
        ensemble_state = snapshot_event.snapshot.get("status")
        final_event_was_EETerminated = False
        snapshot_event_received = False
        async for event in monitor.track():
            if isinstance(event, EETerminated):
                assert snapshot_event_received is True
                final_event_was_EETerminated = True
                assert ensemble_state == ENSEMBLE_STATE_STOPPED
            if type(event) in {EESnapshot, EESnapshotUpdate}:
                # if we get an snapshot event than this need to be valid
                assert final_event_was_EETerminated is False
                snapshot_event_received = True
                if "reals" in event.snapshot:
                    assert ensemble_state == ENSEMBLE_STATE_STARTED
                ensemble_state = event.snapshot.get("status", ensemble_state)
        assert final_event_was_EETerminated is True
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
    if duration > 30 and cpu_seconds / duration > num_cpu * 1.05:
        assert "Misconfigured NUM_CPU" in message
    else:
        assert "NUM_CPU" not in message


@pytest.mark.integration_test
async def test_snapshot_on_resubmit_is_cleared(evaluator_to_use):
    evaluator = evaluator_to_use
    evaluator._batching_interval = 0.4
    token = evaluator._config.token
    url = evaluator._config.get_uri()

    async with Monitor(url, token) as monitor:
        events = monitor.track()

        snapshot_event = await anext(events)
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
            event = await anext(events)
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
            event = await anext(events)
            snapshot = EnsembleSnapshot.from_nested_dict(event.snapshot)
            assert snapshot.get_fm_step("0", "0")["status"] == FORWARD_MODEL_STATE_INIT
            assert snapshot.get_fm_step("0", "1")["status"] == FORWARD_MODEL_STATE_INIT

        await monitor.signal_done()


@pytest.mark.integration_test
async def test_signal_cancel_does_not_cause_evaluator_dispatcher_communication_to_hang(
    evaluator_to_use, monkeypatch
):
    evaluator = evaluator_to_use
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
    async with Monitor(url, token) as monitor:
        async with Client(url, token=token) as dispatch:
            event = ForwardModelStepRunning(
                ensemble=evaluator.ensemble.id_,
                real="0",
                fm_step="0",
                current_memory_usage=1000,
            )
            await dispatch.send(event_to_json(event))

            async def try_sending_event_from_dispatcher_while_monitor_is_cancelling_ensemble():  # noqa: E501
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
                    monitor.signal_cancel(),
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
        async for event in monitor.track():
            final_snapshot.update_from_event(event)
            if is_completed_snapshot(final_snapshot):
                was_completed = True
                break

        assert was_completed


async def test_log_forward_model_steps_with_missing_status_updates(
    monkeypatch, tmpdir, caplog, make_ensemble
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
    evaluator = EnsembleEvaluator(ensemble, mocked_config)
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
