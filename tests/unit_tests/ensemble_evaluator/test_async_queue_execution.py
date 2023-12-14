import asyncio
import threading
from http import HTTPStatus

import pytest
from cloudevents.http import from_json
from websockets.server import serve

from ert.async_utils import get_event_loop
from ert.ensemble_evaluator._wait_for_evaluator import wait_for_evaluator
from ert.job_queue import Driver, JobQueue
from ert.scheduler import Scheduler, create_driver
from ert.shared.feature_toggling import FeatureToggling


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
            if cloud_event["type"] == "com.equinor.ert.realization.success":
                break

    async with serve(_handler, host, port, process_request=process_request):
        await done
    return events


@pytest.mark.asyncio
@pytest.mark.timeout(60)
@pytest.mark.scheduler
async def test_happy_path(
    tmpdir,
    unused_tcp_port,
    event_loop,
    make_ensemble_builder,
    queue_config,
    caplog,
    monkeypatch,
    try_queue_and_scheduler,
):
    asyncio.set_event_loop(event_loop)
    host = "localhost"
    url = f"ws://{host}:{unused_tcp_port}"

    done = get_event_loop().create_future()
    mock_ws_task = get_event_loop().create_task(mock_ws(host, unused_tcp_port, done))
    await wait_for_evaluator(base_url=url, timeout=5)

    ensemble = make_ensemble_builder(monkeypatch, tmpdir, 1, 1).build()

    if FeatureToggling.is_enabled("scheduler"):
        queue = Scheduler(
            create_driver(queue_config), ensemble.reals, ee_uri=url, ens_id="ee_0"
        )
    else:
        queue = JobQueue(queue_config, ensemble.reals, ee_uri=url, ens_id="ee_0")

    await queue.execute()

    done.set_result(None)

    await mock_ws_task

    mock_ws_task.result()

    assert mock_ws_task.done()

    if FeatureToggling.is_enabled("scheduler"):
        first_expected_queue_event_type = "SUBMITTED"
    else:
        first_expected_queue_event_type = "WAITING"

    for received_event, expected_type, expected_queue_event_type in zip(
        [mock_ws_task.result()[0], mock_ws_task.result()[-1]],
        ["waiting", "success"],
        [first_expected_queue_event_type, "SUCCESS"],
    ):
        assert from_json(received_event)["source"] == "/ert/ensemble/ee_0/real/0"
        assert (
            from_json(received_event)["type"]
            == f"com.equinor.ert.realization.{expected_type}"
        )
        assert from_json(received_event).data == {
            "queue_event_type": expected_queue_event_type
        }
