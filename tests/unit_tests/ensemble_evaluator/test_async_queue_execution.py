from http import HTTPStatus

import orjson
import pytest
from websockets.server import serve

from _ert.async_utils import get_running_loop
from ert.ensemble_evaluator._wait_for_evaluator import wait_for_evaluator
from ert.scheduler import Scheduler, create_driver


async def mock_ws(host, port, done):
    events = []

    async def process_request(path, request_headers):
        if path == "/healthcheck":
            return HTTPStatus.OK, {}, b""

    async def _handler(websocket, path):
        while True:
            event = await websocket.recv()
            events.append(event)
            event = orjson.loads(event)
            if event["type"] == "com.equinor.ert.realization.success":
                break

    async with serve(_handler, host, port, process_request=process_request):
        await done
    return events


@pytest.mark.asyncio
@pytest.mark.timeout(60)
async def test_happy_path(
    tmpdir,
    unused_tcp_port,
    make_ensemble,
    queue_config,
    monkeypatch,
):
    host = "localhost"
    url = f"ws://{host}:{unused_tcp_port}"

    done = get_running_loop().create_future()
    mock_ws_task = get_running_loop().create_task(mock_ws(host, unused_tcp_port, done))
    await wait_for_evaluator(base_url=url, timeout=5)

    ensemble = make_ensemble(monkeypatch, tmpdir, 1, 1)

    queue = Scheduler(
        create_driver(queue_config), ensemble.reals, ee_uri=url, ens_id="ee_0"
    )

    await queue.execute()

    done.set_result(None)

    await mock_ws_task

    mock_ws_task.result()

    assert mock_ws_task.done()

    first_expected_queue_event_type = "WAITING"

    for received_event, expected_type, expected_queue_event_type in zip(
        [mock_ws_task.result()[0], mock_ws_task.result()[-1]],
        ["waiting", "success"],
        [first_expected_queue_event_type, "SUCCESS"],
    ):
        event = orjson.loads(received_event)
        assert event["ensemble"] == "ee_0"
        assert event["real"] == 0
        assert event["type"] == f"com.equinor.ert.realization.{expected_type}"
        assert event["data"] == {"queue_event_type": expected_queue_event_type}
