import websockets
import asyncio
import threading


class Client:
    def __enter__(self):
        self.thread.start()
        self.exception = None
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
                async with websockets.connect(self.url) as websocket:
                    while True:
                        msg = await q.get()
                        if msg == "stop":
                            return
                        await websocket.send(msg)

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

