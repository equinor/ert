import asyncio
from pathlib import Path

import aiofiles
import pytest
import websockets
from ert_shared.ensemble_evaluator.nfs_adaptor import nfs_adaptor
from ert_shared.ensemble_evaluator.entity import _FM_STEP_SUCCESS


async def mock_writer(filename, times=2):
    async with aiofiles.open(filename, mode="a") as f:
        for r in range(0, times):
            await f.write(f"Time {r}\n")
            await asyncio.sleep(0.2)
        await f.write(_FM_STEP_SUCCESS)


async def mock_ws(host, port):
    done = asyncio.get_event_loop().create_future()
    lines = []

    async def _handler(websocket, path):
        while True:
            line = await websocket.recv()
            lines.append(line)
            if line == _FM_STEP_SUCCESS:
                done.set_result(None)
                break

    async with websockets.serve(_handler, host, port):
        await done
    return lines[:-1]


@pytest.mark.asyncio
async def test_append_to_file(tmpdir, unused_tcp_port):
    host = "localhost"
    port = unused_tcp_port
    log_file = Path(tmpdir) / "log"

    futures = (
        nfs_adaptor(log_file, f"ws://{host}:{port}"),
        asyncio.get_event_loop().create_task(mock_ws(host, port)),
        mock_writer(log_file),
    )
    done, _ = await asyncio.wait(
        futures, timeout=2, return_when=asyncio.FIRST_EXCEPTION
    )

    exceptions = []
    for t in done:
        if t.exception():
            exceptions.append(t.exception())
        if t == futures[1]:  # mock_ws task
            assert t.result() == ["Time 0\n", "Time 1\n"]
    assert exceptions == []
    assert len(done) == 3
