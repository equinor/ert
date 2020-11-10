import websockets
from websocket import WebSocketException
import asyncio
import threading


class Client:
    def __enter__(self, max_retries=10):
        self.thread.start()
        self.exception = None
        self._max_retries = max_retries
        return self

    def __exit__(self, exc_type, exc_value, exc_traceback):
        self.stop()

    def __init__(self, url):
        self.url = url
        self.loop = asyncio.new_event_loop()
        self.q = asyncio.Queue(loop=self.loop)
        self.thread = threading.Thread(
            name="test_websocket_client", target=self._run, args=(self.loop,)
        )

    def _run(self, loop):
        try:
            asyncio.set_event_loop(loop)
            async def send_loop(q):
                msg = None
                retries = 0
                while True:
                    try:
                        async with websockets.connect(self.url) as websocket:
                            retries = 0
                            while True:
                                if msg is None:
                                    msg = await q.get()
                                if msg == "stop":
                                    return
                                await websocket.send(msg)
                                msg = None
                    except (ConnectionRefusedError, WebSocketException):
                        if retries == self._max_retries:
                            raise
                        await asyncio.sleep(0.2 + 5 * retries)
                        retries += 1

            loop.run_until_complete(send_loop(self.q))
        except Exception as e:
            self.exception = e
            raise

    def send(self, msg):
        self.loop.call_soon_threadsafe(self.q.put_nowait, msg)

    def stop(self):
        self.loop.call_soon_threadsafe(self.q.put_nowait, "stop")
        self.thread.join()
        if self.exception:
            raise Exception("Could not send message") from self.exception

