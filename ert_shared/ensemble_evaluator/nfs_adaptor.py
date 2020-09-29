import asyncio
import os

import aiofiles
import websockets

END_OF_TRANSMISSION = chr(4) + "\n"


class NFSAdaptor:
    def __init__(self, log_file, ws_url):
        self._ws_url = ws_url
        self._log_file = log_file

    async def run(self):
        retries = 0
        max_retries = 3
        while retries < max_retries:
            try:
                await self._run()
                return
            except ConnectionRefusedError as e:
                print(f"{__name__} failed to connect ({retries}/{max_retries}: {e}")
                retries += 1
                await asyncio.sleep(0.2)

    async def _run(self):
        async with websockets.connect(self._ws_url) as websocket:
            async with aiofiles.open(str(self._log_file), "r") as f:
                await f.seek(0, os.SEEK_END)

                line = None
                while line != END_OF_TRANSMISSION:
                    line = await f.readline()
                    if not line:
                        await asyncio.sleep(1)
                        continue
                    await websocket.send(line)
