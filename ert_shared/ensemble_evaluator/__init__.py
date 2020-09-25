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
        self._events = []

        self._ws_thread = threading.Thread(
            name="ert_ee_wsocket",
            target=self._wsocket,
        )
        self._loop = asyncio.new_event_loop()
        self._done = self._loop.create_future()

    def _wsocket(self):
        # loop = asyncio.new_event_loop()
        loop = self._loop
        asyncio.set_event_loop(loop)

        USERS = set()

        # done = loop.create_future()
        done = self._done

        async def worker(done):
            print("Worker started")
            i = 1
            work_is_done = False
            while not (done.done() or work_is_done):
                await asyncio.sleep(1)
                if USERS:
                    event_data = ee_entity.create_evaluator_event(i, None, None, "running")
                    message = json.dumps(event_data.to_dict())
                    await asyncio.wait([user.send(message) for user in USERS])
                i += 1

                if i == 15:
                    print("Worker done")
                    work_is_done = True

            if USERS:
                status = "done" if work_is_done else "terminated"
                event_data = ee_entity.create_evaluator_event(
                    i, None, None, status=status
                )
                message = json.dumps(event_data.to_dict())
                await asyncio.wait([user.send(message) for user in USERS])
            done.set_result(None)
            print("worker exiting")

        async def client_handler(websocket, path):
            print(f"Client {websocket.remote_address} connected")
            USERS.add(websocket)

            snapshot = ee_entity.create_evaluator_event(0, None, None, "running")
            message = json.dumps(snapshot.to_dict())
            await websocket.send(message)

            try:
                async for message in websocket:
                    data = json.loads(message)
                    client_event = ee_entity.create_command_from_dict(data)
                    if client_event.is_terminate():
                        print(
                            f"Client {websocket.remote_address} asked to terminate"
                        )
                        done.set_result(None)
            except Exception as e:
                print(e, traceback.format_exc())
            finally:
                print("Serverside: Client disconnected")
                USERS.remove(websocket)

        async def evaluator_server(done):
            async with websockets.serve(client_handler, self._host, self._port):
                await done
                print("server got done signal")
            print("server exiting")

        worker_future = loop.create_task(worker(done))
        server_future = loop.create_task(evaluator_server(done))
        loop.run_until_complete(asyncio.wait((server_future, worker_future)))

        print("Done")

    def run(self):
        self._ws_thread.start()
        return ee_monitor.create(self._host, self._port)

    def stop(self):
        try:
            self._done.set_result(None)
        except asyncio.exceptions.InvalidStateError:
            pass
