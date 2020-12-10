import asyncio
from pathlib import Path

import aiofiles
from cloudevents.http.event import CloudEvent
from cloudevents.http.json_methods import to_json
import pytest
import websockets
from ert_shared.ensemble_evaluator.nfs_adaptor import nfs_adaptor
from ert_shared.ensemble_evaluator.ws_util import wait as wait_for_ws
from ert_shared.ensemble_evaluator.entity.identifiers import EVTYPE_FM_STEP_SUCCESS


async def mock_writer(filename, times=2):
    async with aiofiles.open(filename, mode="a") as f:
        for r in range(0, times):
            e = CloudEvent(
                {
                    "source": "/mock",
                    "id": f"time-{r}",
                    "type": "fake",
                    "data": {"time": r},
                }
            )
            await f.write(to_json(e).decode() + "\n")
            await asyncio.sleep(0.2)
        await f.write(EVTYPE_FM_STEP_SUCCESS)


async def mock_ws(host, port):
    done = asyncio.get_event_loop().create_future()
    lines = []

    async def _handler(websocket, path):
        while True:
            line = await websocket.recv()
            lines.append(line)
            if line == EVTYPE_FM_STEP_SUCCESS:
                done.set_result(None)
                break

    async with websockets.serve(_handler, host, port):
        await done
    return lines[:-1]


@pytest.mark.asyncio
@pytest.mark.timeout(60)
async def test_append_to_file(tmpdir, unused_tcp_port, event_loop, caplog):
    host = "localhost"
    url = f"ws://{host}:{unused_tcp_port}"
    log_file = Path(tmpdir) / "log"
    mock_ws_task = event_loop.create_task(mock_ws(host, unused_tcp_port))
    await wait_for_ws(url, 10)

    adaptor_task = event_loop.create_task(nfs_adaptor(log_file, url))
    mock_writer_task = event_loop.create_task(mock_writer(log_file))
    await asyncio.wait(
        (adaptor_task, mock_ws_task, mock_writer_task),
        timeout=3,
        return_when=asyncio.FIRST_EXCEPTION,
    )
    adaptor_task.result()
    mock_writer_task.result()
    assert "time-0" in mock_ws_task.result()[0]
    assert "time-1" in mock_ws_task.result()[1]
