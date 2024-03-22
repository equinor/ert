import asyncio
import time
from functools import partial

import pytest

from _ert_job_runner.client import Client
from ert.async_utils import new_event_loop
from ert.ensemble_evaluator import EnsembleEvaluatorAsync, Snapshot, identifiers
from ert.ensemble_evaluator.monitor import Monitor
from ert.ensemble_evaluator.state import (
    ENSEMBLE_STATE_FAILED,
    ENSEMBLE_STATE_STARTED,
    ENSEMBLE_STATE_UNKNOWN,
    FORWARD_MODEL_STATE_FAILURE,
    FORWARD_MODEL_STATE_FINISHED,
    FORWARD_MODEL_STATE_RUNNING,
)

from .ensemble_evaluator_utils import TestEnsemble, send_dispatch_event


@pytest.fixture(name="evaluator_async")
def make_evaluator_async(make_ee_config):
    ensemble = TestEnsemble(0, 2, 2, id_="0")
    ee = EnsembleEvaluatorAsync(
        ensemble,
        make_ee_config(),
        0,
    )
    yield ee
    ee.stop()


async def mock_failure(message, *args, **kwargs):
    raise RuntimeError(message)


@pytest.mark.parametrize(
    ("task, error_msg"),
    [
        ("_dispatcher", "Dispatcher failed!"),
        ("_process_buffer", "Batch processing failed!"),
        ("_publisher", "Publisher failed!"),
        ("_server", "Server failed!"),
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


async def test_dispatch_endpoint_clients_can_connect_and_monitor_can_shut_down_evaluator(
    evaluator_async,
):
    run_task = asyncio.create_task(
        evaluator_async.run_and_get_successful_realizations()
    )
    await evaluator_async._server_started.wait()
    conn_info = evaluator_async._config.get_connection_info()
    with Monitor(conn_info) as monitor:
        events = monitor.track()
        token = evaluator_async._config.token
        cert = evaluator_async._config.cert

        url = evaluator_async._config.url
        # first snapshot before any event occurs
        snapshot_event = next(events)
        snapshot = Snapshot(snapshot_event.data)
        assert snapshot.status == ENSEMBLE_STATE_UNKNOWN
        # two dispatch endpoint clients connect
        with Client(
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
            send_dispatch_event(
                dispatch1,
                identifiers.EVTYPE_FORWARD_MODEL_RUNNING,
                f"/ert/ensemble/{evaluator_async.ensemble.id_}/real/0/forward_model/0",
                "event1",
                {"current_memory_usage": 1000},
            )

            # second dispatch endpoint client informs that job 0 is running
            send_dispatch_event(
                dispatch2,
                identifiers.EVTYPE_FORWARD_MODEL_RUNNING,
                f"/ert/ensemble/{evaluator_async.ensemble.id_}/real/1/forward_model/0",
                "event1",
                {"current_memory_usage": 1000},
            )

            # second dispatch endpoint client informs that job 0 is done
            send_dispatch_event(
                dispatch2,
                identifiers.EVTYPE_FORWARD_MODEL_SUCCESS,
                f"/ert/ensemble/{evaluator_async.ensemble.id_}/real/1/forward_model/0",
                "event1",
                {"current_memory_usage": 1000},
            )

            # second dispatch endpoint client informs that job 1 is failed
            send_dispatch_event(
                dispatch2,
                identifiers.EVTYPE_FORWARD_MODEL_FAILURE,
                f"/ert/ensemble/{evaluator_async.ensemble.id_}/real/1/forward_model/1",
                "event_job_1_fail",
                {identifiers.ERROR_MSG: "error"},
            )
            evt = next(events)
            snapshot = Snapshot(evt.data)
            assert snapshot.get_job("1", "0").status == FORWARD_MODEL_STATE_FINISHED
            assert snapshot.get_job("0", "0").status == FORWARD_MODEL_STATE_RUNNING
            assert snapshot.get_job("1", "1").status == FORWARD_MODEL_STATE_FAILURE

        # a second monitor connects
        with Monitor(evaluator_async._config.get_connection_info()) as monitor2:
            events2 = monitor2.track()
            full_snapshot_event = next(events2)
            assert full_snapshot_event["type"] == identifiers.EVTYPE_EE_SNAPSHOT
            snapshot = Snapshot(full_snapshot_event.data)
            assert snapshot.status == ENSEMBLE_STATE_UNKNOWN
            assert snapshot.get_job("1", "0").status == FORWARD_MODEL_STATE_FINISHED
            assert snapshot.get_job("0", "0").status == FORWARD_MODEL_STATE_RUNNING
            assert snapshot.get_job("1", "1").status == FORWARD_MODEL_STATE_FAILURE

            # one monitor requests that server exit
            monitor.signal_cancel()

            # both monitors should get a terminated event
            terminated = next(events)
            terminated2 = next(events2)
            assert terminated["type"] == identifiers.EVTYPE_EE_TERMINATED
            assert terminated2["type"] == identifiers.EVTYPE_EE_TERMINATED

            for e in [events, events2]:
                for unexpected_event in e:
                    raise AssertionError(
                        f"got unexpected event {unexpected_event} from monitor"
                    )

    await run_task
