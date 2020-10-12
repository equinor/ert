import asyncio
import json
import time

import aiofiles
import websockets
from ert_shared.ensemble_evaluator.ws_util import wait

END_OF_TRANSMISSION = chr(4) + "\n"


class NFSAdaptor:
    def __init__(self, log_file, ws_url, max_batch_size=1000, batch_timeout=2):
        self._ws_url = ws_url
        self._log_file = log_file
        self._max_batch_size = max_batch_size
        self._batch_timeout = batch_timeout

    async def run(self):
        await wait(self._ws_url, 3)
        await self._run()

    async def _run(self):
        async with websockets.connect(self._ws_url) as websocket:
            async with aiofiles.open(str(self._log_file), "r") as f:
                await _batch_lines(
                    f, websocket, self._max_batch_size, self._batch_timeout
                )


async def _batch_lines(fh, ws, batch_size, batch_timeout):
    queue = asyncio.Queue()
    producer = asyncio.create_task(_produce(queue, fh))
    batcher = asyncio.create_task(_batch(queue, ws, batch_size, batch_timeout))
    await asyncio.wait((producer, batcher), return_when=asyncio.ALL_COMPLETED)
    batcher.result()
    producer.result()
    await queue.join()


async def _produce(queue, fh):
    line = None
    while line != END_OF_TRANSMISSION:
        line = await fh.readline()
        if not line:
            await asyncio.sleep(1)
            continue
        await queue.put(line)


async def _batch(queue, ws, batch_size, timeout):
    batch = []
    start = time.time()
    item = None
    while item != END_OF_TRANSMISSION:
        item = await queue.get()
        batch.append(item)

        assert item is not None

        if (
            len(batch) >= batch_size
            or (time.time() - start) > timeout
            or item == END_OF_TRANSMISSION
        ):
            if len(batch) > 0:
                await ws.send(json.dumps(batch))

            batch = []
            start = time.time()

        queue.task_done()
