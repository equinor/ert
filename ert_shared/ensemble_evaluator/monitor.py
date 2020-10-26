import asyncio
import websockets
from ert_shared.ensemble_evaluator.ws_util import wait_for_ws
import queue
import logging
import threading
from cloudevents.http import from_json
from cloudevents.http.event import CloudEvent
from cloudevents.http import to_json
import ert_shared.ensemble_evaluator.entity.identifiers as identifiers

logger = logging.getLogger(__name__)


class _Monitor:
    def __init__(self, host, port):
        self._host = host
        self._port = port
        self._loop = asyncio.new_event_loop()
        self._outbound = asyncio.Queue(loop=self._loop)
        self._receive_future = None
        self._event_index = 1

    def event_index(self):
        index = self._event_index
        self._event_index += 1
        return index

    def exit_server(self):
        out_cloudevent = CloudEvent(
            {
                "type": identifiers.EVTYPE_EE_TERMINATE_REQUEST,
                "source": "/ert/monitor/0",
                "id": self.event_index(),
            }
        )
        message = to_json(out_cloudevent)
        self._loop.call_soon_threadsafe(self._outbound.put_nowait(message))
        logger.debug(f"sent message {message}")

    def track(self):
        incoming = asyncio.Queue(loop=self._loop)
        monitor = self

        async def send(websocket):
            while True:
                msg = await monitor._outbound.get()
                logger.debug("monitor sending:" + msg.decode())
                await websocket.send(msg)

        async def receive():
            logger.debug("starting monitor receive")
            uri = f"ws://{self._host}:{self._port}/client"
            async with websockets.connect(
                uri, max_size=2 ** 26, max_queue=500
            ) as websocket:
                send_future = asyncio.Task(send(websocket), loop=self._loop)
                try:
                    async for message in websocket:
                        logger.debug(f"monitor receive: {message}")
                        event = from_json(message)
                        incoming.put_nowait(event)
                        if event["type"] == identifiers.EVTYPE_EE_TERMINATED:
                            logger.debug("client received terminated")
                            break
                except Exception as e:
                    import traceback

                    logger.error(e, traceback.format_exc())
                finally:
                    logger.debug("cancel send")
                    send_future.cancel()
                    await send_future
            logger.debug("monitor disconnected")

        wait_for_ws(f"ws://{self._host}:{self._port}")

        def run(done_future):
            asyncio.set_event_loop(monitor._loop)
            monitor._receive_future = monitor._loop.create_task(receive())
            try:
                monitor._loop.run_until_complete(monitor._receive_future)
            except asyncio.CancelledError:
                logger.debug("receive cancelled")
            monitor._loop.run_until_complete(done_future)

        done_future = asyncio.Future(loop=self._loop)

        thread = threading.Thread(
            name="ert_monitor_loop", target=run, args=(done_future,)
        )
        thread.start()

        running = True
        try:
            while running:
                logger.debug("wait for incoming")
                event = asyncio.run_coroutine_threadsafe(
                    incoming.get(), self._loop
                ).result()
                logger.debug(f"got incoming: {event}")
                if event["type"] == identifiers.EVTYPE_EE_TERMINATED:
                    running = False
                    done_future.set_result(None)
                yield event
        except GeneratorExit:
            logger.debug("generator exit")
            self._loop.call_soon_threadsafe(self._receive_future.cancel)
            if not done_future.done():
                done_future.set_result(None)
        thread.join()


def create(host, port):
    return _Monitor(host, port)
