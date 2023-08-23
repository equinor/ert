import asyncio
import threading
from http import HTTPStatus

import pytest
from cloudevents.http import from_json
from websockets.server import serve

from ert.async_utils import get_event_loop
from ert.ensemble_evaluator._wait_for_evaluator import wait_for_evaluator
from ert.job_queue import Driver, JobQueue


async def mock_ws(host, port, done):
    events = []

    async def process_request(path, request_headers):
        if path == "/healthcheck":
            return HTTPStatus.OK, {}, b""

    async def _handler(websocket, path):
        while True:
            event = await websocket.recv()
            events.append(event)
            cloud_event = from_json(event)
            if cloud_event["type"] == "com.equinor.ert.forward_model_stage.success":
                break

    async with serve(_handler, host, port, process_request=process_request):
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
    queue = JobQueue(
        Driver.create_driver(queue_config), max_submit=queue_config.max_submit
    )
    for real in ensemble.reals:
        queue.add_ee_stage(real.steps[0], None)

    await queue.execute_queue_via_websockets(
        url, "ee_0", threading.BoundedSemaphore(value=10), None
    )
    done.set_result(None)

    await mock_ws_task

    mock_ws_task.result()

    assert mock_ws_task.done()

    event_0 = from_json(mock_ws_task.result()[0])
    assert event_0["source"] == "/ert/ensemble/ee_0/real/0/step/0"
    assert event_0["type"] == "com.equinor.ert.forward_model_step.waiting"
    assert event_0.data == {"queue_event_type": "WAITING"}

    end_event_index = len(mock_ws_task.result()) - 1
    end_event = from_json(mock_ws_task.result()[end_event_index])
    assert end_event["type"] == "com.equinor.ert.forward_model_step.success"
    assert end_event.data == {"queue_event_type": "SUCCESS"}
