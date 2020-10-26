import asyncio
import os.path
import aiofiles
import websockets
from ert_shared.ensemble_evaluator.entity.identifiers import (
    EVTYPE_FM_STEP_FAILURE,
    EVTYPE_FM_STEP_SUCCESS,
)
from ert_shared.ensemble_evaluator.ws_util import wait


async def _wait_for_filepath(filepath):
    while not os.path.isfile(filepath):
        await asyncio.sleep(0.5)


async def nfs_adaptor(log_file, ws_url):
    await _wait_for_filepath(log_file)
    await wait(ws_url, 25)
    async with websockets.connect(ws_url) as websocket:
        async with aiofiles.open(str(log_file), "r") as f:
            line = None
            while not _is_end_event(line):
                line = await f.readline()
                if not line:
                    await asyncio.sleep(1)
                    continue
                line = line[:-1] if line[-1:] == chr(10) else line
                await websocket.send(line)


def _is_end_event(line):
    return line is not None and (
        EVTYPE_FM_STEP_FAILURE in line or EVTYPE_FM_STEP_SUCCESS in line
    )
