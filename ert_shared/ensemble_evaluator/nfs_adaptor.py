import asyncio

import aiofiles
import websockets
from ert_shared.ensemble_evaluator.ws_util import wait

END_OF_TRANSMISSION = chr(4) + "\n"


class NFSAdaptor:
    def __init__(self, log_file, ws_url):
        self._ws_url = ws_url
        self._log_file = log_file

    async def run(self):
        await wait(self._ws_url, 25)
        await self._run()

    async def _run(self):
        async with websockets.connect(self._ws_url) as websocket:
            async with aiofiles.open(str(self._log_file), "r") as f:
                line = None
                while line != END_OF_TRANSMISSION:
                    line = await f.readline()
                    if not line:
                        await asyncio.sleep(1)
                        continue
                    await websocket.send(line)
