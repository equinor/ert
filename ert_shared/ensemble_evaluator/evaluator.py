from ert_gui.plottery.plots import ensemble
import threading
from ert_shared.ensemble_evaluator.entity import RealizationDecoder, create_evaluator_event, create_forward_model_job, create_realization, create_unindexed_evaluator_event
import ert_shared.ensemble_evaluator.monitor as ee_monitor
import ert_shared.ensemble_evaluator.entity as ee_entity
import json
import asyncio
import websockets
from cloudevents.http import from_json


class EnsembleEvaluator:
    def __init__(self, ensemble, port=8765):

        self._host = "localhost"
        self._port = port
        self._ensemble = ensemble

        self._ws_thread = threading.Thread(
            name="ert_ee_wsocket",
            target=self._wsocket,
        )
        self._loop = asyncio.new_event_loop()
        self._done = self._loop.create_future()

        self._snapshot = ensemble.snapshot()
        ensemble.evaluate(self._host, self._port, "ee_id")

    def _wsocket(self):
        loop = self._loop
        asyncio.set_event_loop(loop)

        USERS = set()
        done = self._done

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
                        self._stop()
            except Exception as e:
                import traceback

                print(e, traceback.format_exc())
            finally:
                print("Serverside: Client disconnected")
                USERS.remove(websocket)

        async def handle_dispatch(websocket, path):
            print(path)
            # dispatch_id = int(path.split("/")[2])  # assuming format is /dispatch/<id>
            # event_data = ee_entity.create_evaluator_event(dispatch_id, None, "started")
            # message = json.dumps(event_data.to_dict())
            # await asyncio.wait([user.send(message) for user in USERS])
            try:
                async for msg in websocket:
                    print(f"dispatch got: {msg}")
                    
                    event = from_json(msg)
                    snapshot_mutate_event = None
                    real_id = get_real_id(event["source"])
                    stage_id = get_stage_id(event["source"])
                    if event["type"] == ee_entity._FM_JOB_RUNNING:
                        step_id = get_step_id(event["source"])
                        job_id = get_job_id(event["source"])
                        self._snapshot._realizations[real_id]["forward_models"][job_id]["status"] == "running"
                        # self._snapshot._realizations[real_id]["forward_models"][job_id]["status"] == "running"
                        snapshot_mutate_event = {
                            real_id: {
                                stage_id: {
                                    step_id: {
                                        job_id: {
                                            "status": "running"
                                        }
                                    }
                                }
                            }
                        }
                    self._snapshot.merge_event(snapshot_mutate_event)
                    out_msg = json.dumps(snapshot_mutate_event.to_dict(), cls=RealizationDecoder)
                    if USERS:
                        await asyncio.wait([user.send(out_msg) for user in USERS])
            except Exception as e:
                import traceback
                print(e, traceback.format_exc())
                if USERS:
                    msg = ee_entity.create_evaluator_event(0, None, "terminated")
                    message = json.dumps(msg.to_dict())
                    if USERS:
                        await asyncio.wait([user.send(message) for user in USERS])
            finally:
                print("dispatch exit")

        async def connection_handler(websocket, path):
            print("connection_handler start")
            elements = path.split("/")
            if elements[1] == "client":
                await handle_client(websocket, path)
            elif elements[1] == "dispatch":
                await handle_dispatch(websocket, path)

        async def evaluator_server(done):
            async with websockets.serve(connection_handler, self._host, self._port):
                await done
                print("server got done signal")

                msg = ee_entity.create_evaluator_event(0, None, "terminated")
                message = json.dumps(msg.to_dict())
                if USERS:
                    await asyncio.wait([user.send(message) for user in USERS])
                print("send terminated")
            print("server exiting")

        server_future = loop.create_task(evaluator_server(done))
        loop.run_until_complete(server_future)

        print("Evaluator thread Done")

    def run(self):
        self._ws_thread.start()
        return ee_monitor.create(self._host, self._port)

    def _stop(self):
        if not self._done.done():
            self._done.set_result(None)

    def stop(self):
        self._loop.call_soon_threadsafe(self._stop)


def get_job_id(path):
    return int(path.split("/")[11])


def get_step_id(path):
    return int(path.split("/")[9])


def get_stage_id(path):
    return int(path.split("/")[7])


def get_real_id(path):
    return int(path.split("/")[5])


def get_ee_id(path):
    return int(path.split("/")[3])
