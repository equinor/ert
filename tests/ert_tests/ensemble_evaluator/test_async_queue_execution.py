import asyncio
import threading
from http import HTTPStatus
import logging

import pytest
import websockets
from cloudevents.http import from_json

from ert_shared.async_utils import get_event_loop
from ert_shared.ensemble_evaluator.utils import wait_for_evaluator


async def mock_ws(host, port, done, close_on_first_attempt=False):
    events = []
    been_closed = False

    async def process_request(path, request_headers):
        if path == "/healthcheck":
            return HTTPStatus.OK, {}, b""

    async def _handler(websocket, path):
        nonlocal been_closed
        nonlocal close_on_first_attempt
        if close_on_first_attempt and not been_closed:
            await websocket.close(code=1001, reason="expected close")
            been_closed = True
            return

        async for event in websocket:
            events.append(event)

    async with websockets.serve(_handler, host, port, process_request=process_request):
        await done
    return events


@pytest.mark.asyncio
@pytest.mark.timeout(60)
async def test_happy_path(
    tmpdir, unused_tcp_port, event_loop, make_ensemble_builder, queue_config, caplog
):
    asyncio.set_event_loop(event_loop)
    host = "localhost"
    url = f"ws://{host}:{unused_tcp_port}"

    done = get_event_loop().create_future()
    mock_ws_task = get_event_loop().create_task(mock_ws(host, unused_tcp_port, done))
    await wait_for_evaluator(base_url=url, timeout=5)

    ensemble = make_ensemble_builder(tmpdir, 1, 1).build()
    queue = queue_config.create_job_queue()
    for real in ensemble.get_reals():
        queue.add_ee_stage(real.get_steps()[0], None)
    queue.submit_complete()

    await queue.execute_queue_async(
        url, "ee_0", threading.BoundedSemaphore(value=10), None
    )
    done.set_result(None)

    await mock_ws_task

    mock_ws_task.result()

    assert mock_ws_task.done()

    event_0 = from_json(mock_ws_task.result()[0])
    assert event_0["source"] == "/ert/ee/ee_0/real/0/step/0"
    assert event_0["type"] == "com.equinor.ert.forward_model_step.waiting"
    assert event_0.data == {"queue_event_type": "JOB_QUEUE_WAITING"}

    end_event_index = len(mock_ws_task.result()) - 1
    end_event = from_json(mock_ws_task.result()[end_event_index])
    assert end_event["type"] == "com.equinor.ert.forward_model_step.success"
    assert end_event.data == {"queue_event_type": "JOB_QUEUE_SUCCESS"}


@pytest.mark.asyncio
@pytest.mark.timeout(60)
async def test_retry_on_close(
    tmpdir, unused_tcp_port, event_loop, make_ensemble_builder, queue_config, caplog
):
    caplog.set_level(logging.INFO)
    asyncio.set_event_loop(event_loop)
    host = "localhost"
    url = f"ws://{host}:{unused_tcp_port}"

    done = get_event_loop().create_future()
    mock_ws_task = get_event_loop().create_task(
        mock_ws(
            host,
            unused_tcp_port,
            done,
            close_on_first_attempt=True,
        )
    )
    await wait_for_evaluator(base_url=url, timeout=5)

    ensemble = make_ensemble_builder(tmpdir, 1, 1).build()
    queue = queue_config.create_job_queue()
    for real in ensemble.get_reals():
        queue.add_ee_stage(real.get_steps()[0], None)
    queue.submit_complete()

    await queue.execute_queue_async(
        url,
        "ee_0",
        threading.BoundedSemaphore(value=10),
        None,
    )
    done.set_result(None)

    await mock_ws_task

    mock_ws_task.result()

    assert mock_ws_task.done()

    # Verify that the test inflicts the expected ConnectionClosed exception
    assert any(
        [
            record
            for record in caplog.records
            if "code=1001, reason='expected close'" in str(record.exc_info)
        ]
    )

    # And that we should still be able to reconnect and provide the information
    event_0 = from_json(mock_ws_task.result()[0])
    assert event_0["source"] == "/ert/ee/ee_0/real/0/step/0"
    assert event_0["type"] == "com.equinor.ert.forward_model_step.waiting"
    assert event_0.data == {"queue_event_type": "JOB_QUEUE_WAITING"}

    end_event_index = len(mock_ws_task.result()) - 1
    end_event = from_json(mock_ws_task.result()[end_event_index])
    assert end_event["type"] == "com.equinor.ert.forward_model_step.success"
    assert end_event.data == {"queue_event_type": "JOB_QUEUE_SUCCESS"}
