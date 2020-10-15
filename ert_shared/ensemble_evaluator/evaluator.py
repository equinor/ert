import time
import threading
import traceback
import ert_shared.ensemble_evaluator.monitor as ee_monitor
import ert_shared.ensemble_evaluator.entity as ee_entity
import json
import asyncio
import websockets


class EnsembleEvaluator:
    def __init__(self):

        self._host = "localhost"
        self._port = 8765
        self._state = {}

        self._ws_thread = threading.Thread(
            name="ert_ee_wsocket",
            target=self._wsocket,
        )
        self._loop = asyncio.new_event_loop()
        self._done = self._loop.create_future()

        self._snapshot = ee_entity.create_evaluator_snapshot(
            [
                ee_entity.create_forward_model_job(1, "test1"),
                ee_entity.create_forward_model_job(2, "test2", (1,)),
                ee_entity.create_forward_model_job(3, "test3", (1,)),
                ee_entity.create_forward_model_job(4, "test4", (2, 3)),
            ],
            [0, 1, 3, 4, 5, 9],
        )

    def _wsocket(self):
        loop = self._loop
        asyncio.set_event_loop(loop)
        _dispatcher_queue = asyncio.Queue()

        USERS = set()
        done = self._done

        async def _mock_dispatch(done, dispatcher_queue):
            print("Mock dispatcher started")
            order_realizations = [0, 4, 9, 1, 5, 3]
            order_jobs = [1, 3, 2, 4]
            events = []
            for realization in order_realizations:
                for job in order_jobs:
                    events.append(
                        ee_entity.create_unindexed_evaluator_event(
                            realizations={
                                realization: {
                                    "forward_models": {
                                        job: {
                                            "status": "running",
                                            "data": {"memory": 1000}
                                            if job % 2 == 1 or realization % 2 == 0
                                            else None,
                                        },
                                    },
                                },
                            },
                            status="running",
                        )
                    )
                    events.append(
                        ee_entity.create_unindexed_evaluator_event(
                            realizations={
                                realization: {
                                    "forward_models": {
                                        job: {
                                            "status": "done",
                                        },
                                    },
                                },
                            },
                            status="running",
                        )
                    )
            for event in events:
                await dispatcher_queue.put(event)
                await asyncio.sleep(3)
                if done.done():
                    return

        async def worker(done, dispatcher_queue):
            print("Worker started")
            i = 1
            try:
                while not done.done():
                    try:
                        event = await asyncio.wait_for(
                            dispatcher_queue.get(), timeout=1
                        )
                    except asyncio.TimeoutError:
                        continue
                    event._event_index = i
                    i += 1
                    self._snapshot.merge_event(event)

                    if USERS:
                        message = json.dumps(event.to_dict())
                        await asyncio.wait([user.send(message) for user in USERS])
                    if i == 49:  # Only for dummy data
                        self.stop()
                if USERS:
                    message = json.dumps(self._snapshot.to_dict())
                    await asyncio.wait([user.send(message) for user in USERS])
            except Exception as e:
                import traceback

                print(e, traceback.format_exc())
            self.stop()
            print("worker exiting")

        async def handle_client(websocket, path):
            try:
                print(f"Client {websocket.remote_address} connected")
                USERS.add(websocket)

                message = json.dumps(self._snapshot.to_dict())
                await websocket.send(message)

                async for message in websocket:
                    data = json.loads(message)
                    client_event = ee_entity.create_command_from_dict(data)
                    if client_event.is_terminate():
                        print(f"Client {websocket.remote_address} asked to terminate")
                        self.stop()
            except Exception as e:
                import traceback

                print(e, traceback.format_exc())
            finally:
                print("Serverside: Client disconnected")
                USERS.remove(websocket)

        async def handle_dispatch(websocket, path):
            pass

        async def connection_handler(websocket, path):
            if path == "/client":
                await handle_client(websocket, path)
            elif path == "/dispatch":
                await handle_dispatch(websocket, path)

        async def evaluator_server(done):
            async with websockets.serve(connection_handler, self._host, self._port):
                await done
                print("server got done signal")
            print("server exiting")

        worker_future = loop.create_task(worker(done, _dispatcher_queue))
        server_future = loop.create_task(evaluator_server(done))
        mock_dispatch_future = loop.create_task(_mock_dispatch(done, _dispatcher_queue))
        loop.run_until_complete(
            asyncio.wait((server_future, worker_future, mock_dispatch_future))
        )

        print("Done")

    def run(self):
        self._ws_thread.start()
        return ee_monitor.create(self._host, self._port)

    def stop(self):
        try:
            self._done.set_result(None)
        except asyncio.InvalidStateError:
            pass
