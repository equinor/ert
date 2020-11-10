import aiofiles
import asyncio
import logging
import os.path
import websockets
from ert_shared.ensemble_evaluator.entity.identifiers import (
    EVTYPE_FM_STEP_FAILURE,
    EVTYPE_FM_STEP_SUCCESS,
)

logger = logging.getLogger(__name__)


async def _wait_for_filepath(filepath, attempts_between_report=4):
    attempts = 0
    while True:
        if os.path.isfile(filepath):
            return
        await asyncio.sleep(2)
        if attempts != 0 and attempts % attempts_between_report == 0:
            logger.info(f"Could not find {filepath} after {attempts} attempts")
        attempts += 1


async def nfs_adaptor(log_file, ws_url):
    await _wait_for_filepath(log_file)
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

    logger.debug(f"saw end of {log_file}")


def _is_end_event(line):
    return line is not None and (
        EVTYPE_FM_STEP_FAILURE in line or EVTYPE_FM_STEP_SUCCESS in line
    )
