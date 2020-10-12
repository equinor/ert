import asyncio
import json
from pathlib import Path

import aiofiles
import pytest
import websockets
from ert_shared.ensemble_evaluator.nfs_adaptor import END_OF_TRANSMISSION, NFSAdaptor


async def mock_writer(filename, times):
    async with aiofiles.open(filename, mode="a") as f:
        for r in range(0, times):
            await f.write(f"Time {r}\n")
        await f.write(END_OF_TRANSMISSION)


async def mock_ws(host, port):
    done = asyncio.get_event_loop().create_future()
    lines = []

    async def _handler(websocket, path):
        while True:
            batch_data = await websocket.recv()
            batch = json.loads(batch_data)
            for line in batch:
                lines.append(line)
                if line == END_OF_TRANSMISSION:
                    done.set_result(None)
                    break

    async with websockets.serve(_handler, host, port):
        await done
    return lines[:-1]


@pytest.mark.parametrize(
    "lines",
    [2, 5000],
)
@pytest.mark.asyncio
async def test_integration(tmpdir, lines):
    host = "localhost"
    port = 50001
    log_file = Path(tmpdir) / "log"

    adaptor = asyncio.create_task(NFSAdaptor(log_file, f"ws://{host}:{port}").run())
    ws = asyncio.create_task(mock_ws(host, port))
    writer = asyncio.create_task(mock_writer(log_file, lines))
    _, pending = await asyncio.wait(
        (adaptor, ws, writer), timeout=10, return_when=asyncio.FIRST_EXCEPTION
    )

    assert len(pending) == 0
    adaptor.result()
    writer.result()

    actual_lines = ws.result()
    assert len(actual_lines) == lines
    assert actual_lines[0] == "Time 0\n"
    assert actual_lines[lines - 1] == f"Time {lines-1}\n"
