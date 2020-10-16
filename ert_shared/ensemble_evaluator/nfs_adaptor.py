import asyncio

import aiofiles
import websockets
from ert_shared.ensemble_evaluator.entity import _FM_STEP_FAILURE, _FM_STEP_SUCCESS
from ert_shared.ensemble_evaluator.ws_util import wait


async def nfs_adaptor(log_file, ws_url):
    await wait(ws_url, 25)
    async with websockets.connect(ws_url) as websocket:
        async with aiofiles.open(str(log_file), "r") as f:
            line = None
            while not _is_end_event(line):
                line = await f.readline()
                if not line:
                    await asyncio.sleep(1)
                    continue
                await websocket.send(line)


def _is_end_event(line):
    return line is not None and (_FM_STEP_FAILURE in line or _FM_STEP_SUCCESS in line)
