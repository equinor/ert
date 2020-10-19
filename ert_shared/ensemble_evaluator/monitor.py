import json
import asyncio
import websockets
import ert_shared.ensemble_evaluator.entity as ee_entity
from ert_shared.ensemble_evaluator.ws_util import wait_for_ws
import queue
import threading


class _Monitor:
    def __init__(self, host, port):
        self._host = host
        self._port = port
        self._loop = asyncio.new_event_loop()
        self._outbound = asyncio.Queue(loop=self._loop)
        self._receive_future = None

    def exit_server(self):
        event = ee_entity.create_command_terminate()
        msg = json.dumps(event.to_dict())
        self._loop.call_soon_threadsafe(self._outbound.put_nowait(msg))
        print(f"sent message {msg}")

    def track(self):
        incoming = queue.Queue()
        monitor = self

        async def send(websocket):
            while True:
                msg = await monitor._outbound.get()
                print("monitor sending:" + msg)
                await websocket.send(msg)

        async def receive():
            print("starting monitor receive")
            uri = f"ws://{self._host}:{self._port}/client"
            async with websockets.connect(uri) as websocket:
                send_future = asyncio.Task(send(websocket))
                try:
                    async for message in websocket:
                        print(f"monitor receive: {message}")
                        event_json = json.loads(message)
                        event = ee_entity.create_evaluator_event_from_dict(event_json)
                        self._loop.run_in_executor(None, lambda: incoming.put(event))
                        if event.is_terminated():
                            print("client received terminated")
                            break
                except asyncio.CancelledError:
                    pass
                except Exception as e:
                    import traceback

                    print(e, traceback.format_exc())
                finally:
                    print("cancel send")
                    send_future.cancel()
                    await send_future
            print("monitor disconnected")

        wait_for_ws(f"ws://{self._host}:{self._port}")

        def run():
            asyncio.set_event_loop(monitor._loop)
            monitor._receive_future = monitor._loop.create_task(receive())
            try:
                monitor._loop.run_until_complete(monitor._receive_future)
            except asyncio.CancelledError:
                print("receive cancelled")

        thread = threading.Thread(
            name="ert_monitor_loop",
            target=run,
        )
        thread.start()

        running = True
        try:
            while running:
                print("wait for incoming")
                event = incoming.get()
                print(f"got incoming: {event}")
                if event.is_terminated():
                    running = False
                yield event
        except GeneratorExit:
            print("generator exit")
            self._loop.call_soon_threadsafe(self._receive_future.cancel)
            thread.join()


def create(host, port):
    return _Monitor(host, port)
