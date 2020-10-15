import asyncio
import threading

import pytest
import websockets
from ert_shared.ensemble_evaluator.queue_adaptor import JobQueueManagerAdaptor
from unittest.mock import Mock


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


def mock_queue_mutator(host, port):
    mock_queue = Mock(
        job_list=[Mock(status=Mock(value=4), callback_arguments=[Mock(iens=0)])]
    )
    JobQueueManagerAdaptor.ws_url = f"ws://{host}:{port}"
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

    mutator = threading.Thread(target=mock_queue_mutator, args=(host, port))
    mutator.start()

    done, _ = await asyncio.wait(
        (asyncio.get_event_loop().create_task(mock_ws(host, port)),),
        timeout=2,
        return_when=asyncio.FIRST_EXCEPTION,
    )

    mutator.join()

    assert len(done) == 1
    assert done.pop().result() == [
        '{"0": {"status": "JOB_QUEUE_RUNNING", "forward_models": null}}',
        '{"0": {"status": "JOB_QUEUE_SUCCESS", "forward_models": null}}',
        "null",
    ]
