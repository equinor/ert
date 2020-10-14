import asyncio
import json
import threading
from unittest.mock import Mock

import pytest
import websockets
from cloudevents.http import from_json
from ert_shared.ensemble_evaluator.queue_adaptor import JobQueueManagerAdaptor
from job_runner import JOBS_FILE


async def mock_ws(host, port):
    done = asyncio.get_event_loop().create_future()
    events = []

    async def _handler(websocket, path):
        while True:
            event = await websocket.recv()
            events.append(event)
            if event == "null":
                done.set_result(None)
                break

    async with websockets.serve(_handler, host, port):
        await done
    return events


def mock_queue_mutator(host, port, tmpdir):
    mock_queue = Mock(
        job_list=[
            Mock(
                status=Mock(value=4),
                callback_arguments=[Mock(iens=0)],
                run_path=tmpdir,
            )
        ]
    )
    JobQueueManagerAdaptor.ws_url = f"ws://{host}:{port}"
    JobQueueManagerAdaptor.ee_id = "ee_id_123"
    jm = JobQueueManagerAdaptor(mock_queue)

    mock_queue.job_list[0].status.value = 16  # running
    jm._publish_changes(jm._changes_after_transition())

    mock_queue.job_list[0].status.value = 512  # done
    jm._transition()
    jm._publish_changes(jm._snapshot())
    jm._publish_changes(None)


@pytest.mark.asyncio
async def test_happy_path(tmpdir, unused_tcp_port):
    host = "localhost"
    port = unused_tcp_port

    with open(tmpdir / JOBS_FILE, "w") as jobs_file:
        json.dump({}, jobs_file)

    mutator = threading.Thread(target=mock_queue_mutator, args=(host, port, tmpdir))
    mutator.start()

    mock_ws_task = asyncio.get_event_loop().create_task(mock_ws(host, port))

    await asyncio.wait(
        (mock_ws_task,),
        timeout=2,
        return_when=asyncio.FIRST_EXCEPTION,
    )

    mutator.join()

    assert mock_ws_task.done()

    event_0 = from_json(mock_ws_task.result()[0])
    assert event_0["source"] == "/ert/ee/0/real/0/stage/0"
    assert event_0["type"] == "com.equinor.ert.forward_model_stage.running"
    assert event_0.data == {"queue_event_type": "JOB_QUEUE_RUNNING"}

    event_1 = from_json(mock_ws_task.result()[1])
    assert event_1["type"] == "com.equinor.ert.forward_model_stage.success"
    assert event_1.data == {"queue_event_type": "JOB_QUEUE_SUCCESS"}

    assert mock_ws_task.result()[2] == "null"

    with open(tmpdir / JOBS_FILE, "r") as jobs_file:
        assert json.load(jobs_file) == {
            "ee_id": "ee_id_123",
            "real_id": 0,
            "stage_id": 0,
        }
