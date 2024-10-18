import asyncio
import datetime
from functools import partial
from typing import cast

import pytest
from hypothesis import given
from hypothesis import strategies as st

from _ert.events import (
    EESnapshot,
    EESnapshotUpdate,
    EETerminated,
    EnsembleStarted,
    EnsembleSucceeded,
    ForwardModelStepFailure,
    ForwardModelStepRunning,
    ForwardModelStepSuccess,
    RealizationSuccess,
    event_to_json,
)
from _ert.forward_model_runner.client import Client
from ert.ensemble_evaluator import (
    EnsembleEvaluator,
    EnsembleSnapshot,
    FMStepSnapshot,
    Monitor,
)
from ert.ensemble_evaluator.evaluator import detect_overspent_cpu
from ert.ensemble_evaluator.state import (
    ENSEMBLE_STATE_STARTED,
    ENSEMBLE_STATE_STOPPED,
    ENSEMBLE_STATE_UNKNOWN,
    FORWARD_MODEL_STATE_FAILURE,
    FORWARD_MODEL_STATE_FINISHED,
    FORWARD_MODEL_STATE_RUNNING,
)

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

    evaluator = EnsembleEvaluator(TestEnsemble(0, 2, 2, id_="0"), make_ee_config())
    monkeypatch.setattr(
        EnsembleEvaluator,
        task,
        partial(mock_failure, error_msg),
    )
    with pytest.raises(RuntimeError, match=error_msg):
        await evaluator.run_and_get_successful_realizations()


async def test_no_config_raises_valueerror_when_running():
    evaluator = EnsembleEvaluator(TestEnsemble(0, 2, 2, id_="0"), None)
    with pytest.raises(ValueError, match="no config for evaluator"):
        await evaluator.run_and_get_successful_realizations()


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


async def test_new_connections_are_denied_when_evaluator_is_closing_down(
    evaluator_to_use,
):
    evaluator = evaluator_to_use

    class TestMonitor(Monitor):
        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)
            self._connection_timeout = 1

    async def new_connection():
        await evaluator._server_done.wait()
        async with TestMonitor(evaluator._config.get_connection_info()):
            pass

    new_connection_task = asyncio.create_task(new_connection())
    evaluator.stop()

    with pytest.raises(RuntimeError):
        await new_connection_task


@pytest.fixture(name="evaluator_to_use")
async def evaluator_to_use_fixture(make_ee_config):
    ensemble = TestEnsemble(0, 2, 2, id_="0")
    evaluator = EnsembleEvaluator(ensemble, make_ee_config())
    evaluator._batching_interval = 0.5  # batching can be faster for tests
    run_task = asyncio.create_task(evaluator.run_and_get_successful_realizations())
    await evaluator._server_started.wait()
    yield evaluator
    evaluator.stop()
    await run_task


@pytest.mark.integration_test
@pytest.mark.timeout(20)
async def test_restarted_jobs_do_not_have_error_msgs(evaluator_to_use):
    evaluator = evaluator_to_use
    token = evaluator._config.token
    cert = evaluator._config.cert
    url = evaluator._config.url

    config_info = evaluator._config.get_connection_info()
    async with Monitor(config_info) as monitor:
        # first snapshot before any event occurs
        events = monitor.track()
        snapshot_event = await events.__anext__()
        snapshot = EnsembleSnapshot.from_nested_dict(snapshot_event.snapshot)
        assert snapshot.status == ENSEMBLE_STATE_UNKNOWN
        # two dispatch endpoint clients connect
        async with Client(
            url + "/dispatch",
            cert=cert,
            token=token,
            max_retries=1,
            timeout_multiplier=1,
        ) as dispatch:
            event = ForwardModelStepRunning(
                ensemble=evaluator.ensemble.id_,
                real="0",
                fm_step="0",
                current_memory_usage=1000,
            )
            await dispatch._send(event_to_json(event))

            event = ForwardModelStepFailure(
                ensemble=evaluator.ensemble.id_,
                real="0",
                fm_step="0",
                error_msg="error",
            )
            await dispatch._send(event_to_json(event))

        def is_completed_snapshot(snapshot: EnsembleSnapshot) -> bool:
            try:
                assert (
                    snapshot.get_fm_step("0", "0")["status"]
                    == FORWARD_MODEL_STATE_FAILURE
                )
                assert snapshot.get_fm_step("0", "0")["error"] == "error"
                return True
            except AssertionError:
                return False

        final_snapshot = EnsembleSnapshot()
        async for event in monitor.track():
            final_snapshot.update_from_event(event)
            if is_completed_snapshot(final_snapshot):
                break

    async with Client(
        url + "/dispatch",
        cert=cert,
        token=token,
        max_retries=1,
        timeout_multiplier=1,
    ) as dispatch:
        event = ForwardModelStepSuccess(
            ensemble=evaluator.ensemble.id_,
            real="0",
            fm_step="0",
            current_memory_usage=1000,
        )
        await dispatch._send(event_to_json(event))

    # reconnect new monitor
    async with Monitor(config_info) as new_monitor:

        def check_if_final_snapshot_is_complete(snapshot: EnsembleSnapshot) -> bool:
            try:
                assert snapshot.status == ENSEMBLE_STATE_UNKNOWN
                assert (
                    snapshot.get_fm_step("0", "0")["status"]
                    == FORWARD_MODEL_STATE_FINISHED
                )
                assert not snapshot.get_fm_step("0", "0")["error"]
                return True
            except AssertionError:
                return False

        final_snapshot = EnsembleSnapshot()
        async for event in new_monitor.track():
            final_snapshot = final_snapshot.update_from_event(event)
            if check_if_final_snapshot_is_complete(final_snapshot):
                break


@pytest.mark.integration_test
@pytest.mark.timeout(20)
async def test_new_monitor_can_pick_up_where_we_left_off(evaluator_to_use):
    evaluator = evaluator_to_use

    token = evaluator._config.token
    cert = evaluator._config.cert
    url = evaluator._config.url

    config_info = evaluator._config.get_connection_info()
    async with Monitor(config_info) as monitor:
        async with Client(
            url + "/dispatch",
            cert=cert,
            token=token,
            max_retries=1,
            timeout_multiplier=1,
        ) as dispatch1, Client(
            url + "/dispatch",
            cert=cert,
            token=token,
            max_retries=1,
            timeout_multiplier=1,
        ) as dispatch2:
            # first dispatch endpoint client informs that forward model 0 is running
            event = ForwardModelStepRunning(
                ensemble=evaluator.ensemble.id_,
                real="0",
                fm_step="0",
                current_memory_usage=1000,
            )
            await dispatch1._send(event_to_json(event))
            # second dispatch endpoint client informs that forward model 0 is running
            event = ForwardModelStepRunning(
                ensemble=evaluator.ensemble.id_,
                real="1",
                fm_step="0",
                current_memory_usage=1000,
            )
            await dispatch2._send(event_to_json(event))
            # second dispatch endpoint client informs that forward model 1 is running
            event = ForwardModelStepRunning(
                ensemble=evaluator.ensemble.id_,
                real="1",
                fm_step="1",
                current_memory_usage=1000,
            )
            await dispatch2._send(event_to_json(event))

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
                return True
            except AssertionError:
                return False

        async for event in monitor.track():
            final_snapshot = final_snapshot.update_from_event(event)
            if check_if_all_fm_running(final_snapshot):
                break
        assert final_snapshot.status == ENSEMBLE_STATE_UNKNOWN

        # take down first monitor by leaving context

    async with Client(
        url + "/dispatch",
        cert=cert,
        token=token,
        max_retries=1,
        timeout_multiplier=1,
    ) as dispatch2:
        # second dispatch endpoint client informs that job 0 is done
        event = ForwardModelStepSuccess(
            ensemble=evaluator.ensemble.id_,
            real="1",
            fm_step="0",
            current_memory_usage=1000,
        )
        await dispatch2._send(event_to_json(event))
        # second dispatch endpoint client informs that job 1 is failed
        event = ForwardModelStepFailure(
            ensemble=evaluator.ensemble.id_, real="1", fm_step="1", error_msg="error"
        )
        await dispatch2._send(event_to_json(event))

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
            return True
        except AssertionError:
            return False

    # reconnect new monitor
    async with Monitor(config_info) as new_monitor:
        final_snapshot = EnsembleSnapshot()
        async for event in new_monitor.track():
            final_snapshot = final_snapshot.update_from_event(event)
            if check_if_final_snapshot_is_complete(final_snapshot):
                break


async def test_dispatch_endpoint_clients_can_connect_and_monitor_can_shut_down_evaluator(
    evaluator_to_use,
):
    evaluator = evaluator_to_use
    evaluator._batching_interval = 10

    evaluator._max_batch_size = 4
    conn_info = evaluator._config.get_connection_info()
    async with Monitor(conn_info) as monitor:
        events = monitor.track()
        token = evaluator._config.token
        cert = evaluator._config.cert

        url = evaluator._config.url
        # first snapshot before any event occurs
        snapshot_event = await events.__anext__()
        assert type(snapshot_event) is EESnapshot
        snapshot = EnsembleSnapshot.from_nested_dict(snapshot_event.snapshot)
        assert snapshot.status == ENSEMBLE_STATE_UNKNOWN
        # two dispatch endpoint clients connect
        async with Client(
            url + "/dispatch",
            cert=cert,
            token=token,
            max_retries=1,
            timeout_multiplier=1,
        ) as dispatch1, Client(
            url + "/dispatch",
            cert=cert,
            token=token,
            max_retries=1,
            timeout_multiplier=1,
        ) as dispatch2:
            # first dispatch endpoint client informs that real 0 fm 0 is running
            event = ForwardModelStepRunning(
                ensemble=evaluator.ensemble.id_,
                real="0",
                fm_step="0",
                current_memory_usage=1000,
            )
            await dispatch1._send(event_to_json(event))
            # second dispatch endpoint client informs that real 1 fm 0 is running
            event = ForwardModelStepRunning(
                ensemble=evaluator.ensemble.id_,
                real="1",
                fm_step="0",
                current_memory_usage=1000,
            )
            await dispatch2._send(event_to_json(event))
            # second dispatch endpoint client informs that real 1 fm 0 is done
            event = ForwardModelStepSuccess(
                ensemble=evaluator.ensemble.id_,
                real="1",
                fm_step="0",
                current_memory_usage=1000,
            )
            await dispatch2._send(event_to_json(event))
            # second dispatch endpoint client informs that real 1 fm 1 is failed
            event = ForwardModelStepFailure(
                ensemble=evaluator.ensemble.id_,
                real="1",
                fm_step="1",
                error_msg="error",
            )
            await dispatch2._send(event_to_json(event))

            event = await events.__anext__()
            snapshot = EnsembleSnapshot.from_nested_dict(event.snapshot)
            assert (
                snapshot.get_fm_step("1", "0")["status"] == FORWARD_MODEL_STATE_FINISHED
            )
            assert (
                snapshot.get_fm_step("0", "0")["status"] == FORWARD_MODEL_STATE_RUNNING
            )
            assert (
                snapshot.get_fm_step("1", "1")["status"] == FORWARD_MODEL_STATE_FAILURE
            )
        # a second monitor connects
        async with Monitor(evaluator._config.get_connection_info()) as monitor2:
            events2 = monitor2.track()
            full_snapshot_event = await events2.__anext__()
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
            terminated = await events.__anext__()
            terminated2 = await events2.__anext__()
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

    config_info = evaluator._config.get_connection_info()
    async with Monitor(config_info) as monitor:
        events = monitor.track()

        token = evaluator._config.token
        cert = evaluator._config.cert
        url = evaluator._config.url

        snapshot_event = await events.__anext__()
        assert type(snapshot_event) is EESnapshot
        async with Client(url + "/dispatch", cert=cert, token=token) as dispatch:
            event = EnsembleStarted(ensemble=evaluator.ensemble.id_)
            await dispatch._send(event_to_json(event))
            event = RealizationSuccess(
                ensemble=evaluator.ensemble.id_, real="0", queue_event_type=""
            )
            await dispatch._send(event_to_json(event))
            event = RealizationSuccess(
                ensemble=evaluator.ensemble.id_, real="1", queue_event_type=""
            )
            await dispatch._send(event_to_json(event))
            event = EnsembleSucceeded(ensemble=evaluator.ensemble.id_)
            await dispatch._send(event_to_json(event))

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
                assert snapshot_event_received == True
                final_event_was_EETerminated = True
                assert ensemble_state == ENSEMBLE_STATE_STOPPED
            if type(event) in [EESnapshot, EESnapshotUpdate]:
                # if we get an snapshot event than this need to be valid
                assert final_event_was_EETerminated == False
                snapshot_event_received = True
                if "reals" in event.snapshot:
                    assert ensemble_state == ENSEMBLE_STATE_STARTED
                ensemble_state = event.snapshot.get("status", ensemble_state)
        assert final_event_was_EETerminated == True
        assert snapshot_event_received == True


@given(
    num_cpu=st.integers(min_value=1, max_value=64),
    start=st.datetimes(),
    duration=st.integers(min_value=-1, max_value=10000),
    cpu_seconds=st.floats(min_value=0),
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
    if duration > 0 and cpu_seconds / duration > num_cpu:
        assert "Misconfigured NUM_CPU" in message
    else:
        assert "NUM_CPU" not in message
