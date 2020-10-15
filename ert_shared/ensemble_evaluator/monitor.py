import time
import json
import asyncio
import websockets
import ert_shared.ensemble_evaluator.entity as ee_entity


class _Monitor:
    def __init__(self, host, port):
        self._host = host
        self._port = port

    def track(self):
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)

        async def hello(queue):
            print("starting monitor hello")
            uri = f"ws://{self._host}:{self._port}/client"
            async with websockets.connect(uri) as websocket:
                async for message in websocket:
                    try:
                        event_json = json.loads(message)
                        event = ee_entity.create_evaluator_event_from_dict(event_json)
                        await queue.put(event)
                        if event.is_done():
                            print("hello exiting return")
                            return
                    except Exception as e:
                        import traceback

                        print(e, traceback.format_exc())

        retries = 0
        while retries < 3:
            try:
                queue = asyncio.Queue()
                hello_future = loop.create_task(hello(queue))
                while True:
                    get_future = loop.create_task(queue.get())
                    done, pending = loop.run_until_complete(
                        asyncio.wait(
                            (hello_future, get_future),
                            return_when=asyncio.FIRST_COMPLETED,
                        )
                    )
                    if hello_future in done:
                        if hello_future.exception():
                            raise hello_future.exception()
                        print("monitor client exited", hello_future)
                        # event = asyncio.run_coroutine_threadsafe(queue.get(), loop).result()
                        # if the client exits (e.g. if evaluator exits early), we
                        # must manually drain the queue IF the get_future is still pending
                        if get_future in done:
                            yield get_future.result()
                        break

                    event = get_future.result()
                    yield event
                    if event.is_done():
                        break
                    # if event.is_done():
                    #     print("monitor saw done, exiting")
                    #     return
                    # yield event
                while not queue.empty():
                    yield loop.run_until_complete(queue.get())

                return
            except OSError as e:
                import traceback

                print(traceback.format_exc())
                print(f"Attempt {retries}: {e}")
                retries += 1
                time.sleep(1)


def create(host, port):
    return _Monitor(host, port)
