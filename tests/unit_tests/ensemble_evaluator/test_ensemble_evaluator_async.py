import asyncio
from functools import partial

import pytest

from _ert_job_runner.client import Client
from ert.ensemble_evaluator import EnsembleEvaluatorAsync, Snapshot, identifiers
from ert.ensemble_evaluator.monitor_async import MonitorAsync
from ert.ensemble_evaluator.state import (
    ENSEMBLE_STATE_UNKNOWN,
    FORWARD_MODEL_STATE_FAILURE,
    FORWARD_MODEL_STATE_FINISHED,
    FORWARD_MODEL_STATE_RUNNING,
)

from .ensemble_evaluator_utils import TestEnsemble, send_dispatch_event_async


@pytest.fixture(name="evaluator_async")
async def make_evaluator_async(make_ee_config):
    ensemble = TestEnsemble(0, 2, 2, id_="0")
    ee = EnsembleEvaluatorAsync(
        ensemble,
        make_ee_config(),
        0,
    )
    yield ee
    await ee._stop()


async def mock_failure(message, *args, **kwargs):
    raise RuntimeError(message)


@pytest.mark.parametrize(
    ("task, error_msg"),
    [
        ("_dispatcher", "Dispatcher failed!"),
        ("_process_buffer", "Batch processing failed!"),
        ("_publisher", "Publisher failed!"),
        # ("_server", "Server failed!"),
    ],
)
async def test_when_task_fails_evaluator_raises_exception(
    task, error_msg, evaluator_async, monkeypatch
):
    monkeypatch.setattr(
        EnsembleEvaluatorAsync,
        task,
        partial(mock_failure, error_msg),
    )
    with pytest.raises(RuntimeError, match=error_msg):
        await evaluator_async.run_and_get_successful_realizations()


@pytest.mark.asyncio
async def test_dispatch_endpoint_clients_can_connect_and_monitor_can_shut_down_evaluator(
    evaluator_async,
):
    run_task = asyncio.create_task(
        evaluator_async.run_and_get_successful_realizations()
    )
    await evaluator_async._server_started.wait()
    conn_info = evaluator_async._config.get_connection_info()
    async with MonitorAsync(conn_info) as monitor:
        token = evaluator_async._config.token
        cert = evaluator_async._config.cert

        url = evaluator_async._config.url
        # first snapshot before any event occurs
        # snapshot_event = await next(events)
        snapshot_event = await monitor.get_event()
        snapshot = Snapshot(snapshot_event.data)
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
            # first dispatch endpoint client informs that job 0 is running
            await send_dispatch_event_async(
                dispatch1,
                identifiers.EVTYPE_FORWARD_MODEL_RUNNING,
                f"/ert/ensemble/{evaluator_async.ensemble.id_}/real/0/forward_model/0",
                "event1",
                {"current_memory_usage": 1000},
            )

            # second dispatch endpoint client informs that job 0 is running
            await send_dispatch_event_async(
                dispatch2,
                identifiers.EVTYPE_FORWARD_MODEL_RUNNING,
                f"/ert/ensemble/{evaluator_async.ensemble.id_}/real/1/forward_model/0",
                "event1",
                {"current_memory_usage": 1000},
            )

            # second dispatch endpoint client informs that job 0 is done
            await send_dispatch_event_async(
                dispatch2,
                identifiers.EVTYPE_FORWARD_MODEL_SUCCESS,
                f"/ert/ensemble/{evaluator_async.ensemble.id_}/real/1/forward_model/0",
                "event1",
                {"current_memory_usage": 1000},
            )

            # second dispatch endpoint client informs that job 1 is failed
            await send_dispatch_event_async(
                dispatch2,
                identifiers.EVTYPE_FORWARD_MODEL_FAILURE,
                f"/ert/ensemble/{evaluator_async.ensemble.id_}/real/1/forward_model/1",
                "event_job_1_fail",
                {identifiers.ERROR_MSG: "error"},
            )
            evt = await monitor.get_event()
            snapshot = Snapshot(evt.data)
            assert snapshot.get_job("1", "0").status == FORWARD_MODEL_STATE_FINISHED
            assert snapshot.get_job("0", "0").status == FORWARD_MODEL_STATE_RUNNING
            assert snapshot.get_job("1", "1").status == FORWARD_MODEL_STATE_FAILURE

        # a second monitor connects
        async with MonitorAsync(
            evaluator_async._config.get_connection_info()
        ) as monitor2:
            full_snapshot_event = await monitor2.get_event()
            assert full_snapshot_event["type"] == identifiers.EVTYPE_EE_SNAPSHOT
            snapshot = Snapshot(full_snapshot_event.data)
            assert snapshot.status == ENSEMBLE_STATE_UNKNOWN
            assert snapshot.get_job("1", "0").status == FORWARD_MODEL_STATE_FINISHED
            assert snapshot.get_job("0", "0").status == FORWARD_MODEL_STATE_RUNNING
            assert snapshot.get_job("1", "1").status == FORWARD_MODEL_STATE_FAILURE

            # one monitor requests that server exit
            await monitor.signal_cancel()

            # both monitors should get a terminated event
            terminated = await monitor.get_event()
            terminated2 = await monitor2.get_event()

            assert terminated["type"] == identifiers.EVTYPE_EE_TERMINATED
            assert terminated2["type"] == identifiers.EVTYPE_EE_TERMINATED

    num_realization = await run_task
    assert len(num_realization) == 0


@pytest.mark.asyncio
async def test_ensure_multi_level_events_in_order(evaluator_async):
    run_task = asyncio.create_task(
        evaluator_async.run_and_get_successful_realizations()
    )
    await evaluator_async._server_started.wait()
    async with MonitorAsync(evaluator_async._config.get_connection_info()) as monitor:
        token = evaluator_async._config.token
        cert = evaluator_async._config.cert
        url = evaluator_async._config.url

        snapshot_event = await monitor.get_event()
        assert snapshot_event["type"] == identifiers.EVTYPE_EE_SNAPSHOT
        async with Client(url + "/dispatch", cert=cert, token=token) as dispatch1:
            await send_dispatch_event_async(
                dispatch1,
                identifiers.EVTYPE_ENSEMBLE_STARTED,
                f"/ert/ensemble/{evaluator_async.ensemble.id_}",
                "event0",
                {},
            )
            await send_dispatch_event_async(
                dispatch1,
                identifiers.EVTYPE_REALIZATION_SUCCESS,
                f"/ert/ensemble/{evaluator_async.ensemble.id_}/real/0",
                "event1",
                {},
            )
            await send_dispatch_event_async(
                dispatch1,
                identifiers.EVTYPE_REALIZATION_SUCCESS,
                f"/ert/ensemble/{evaluator_async.ensemble.id_}/real/1",
                "event2",
                {},
            )
            await send_dispatch_event_async(
                dispatch1,
                identifiers.EVTYPE_ENSEMBLE_STOPPED,
                f"/ert/ensemble/{evaluator_async.ensemble.id_}",
                "event3",
                {},
            )
        await monitor.signal_done()
        event = await monitor.get_event()
        assert event["type"] == identifiers.EVTYPE_EE_TERMINATED
        # TODO there is only terminated event? Why?

    await run_task
