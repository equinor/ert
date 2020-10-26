import asyncio
import json
import threading

import ert_shared.ensemble_evaluator.dispatch as dispatch
import ert_shared.ensemble_evaluator.entity.identifiers as identifiers
import ert_shared.ensemble_evaluator.entity.identifiers as ids
import ert_shared.ensemble_evaluator.monitor as ee_monitor
import websockets
from cloudevents.http import from_json, to_json
from cloudevents.http.event import CloudEvent
from ert_shared.ensemble_evaluator.entity.snapshot import (
    SnapshotBuilder,
    PartialSnapshot,
    Snapshot,
)


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

        self._users = set()

        # TODO: This should not be the same
        self._snapshot = ensemble.forward_model_description()
        self._event_index = 1

    @dispatch.register_event_handler(ids.EVGROUP_FM_ALL)
    async def _fm_handler(self, event):
        snapshot_mutate_event = PartialSnapshot.from_cloudevent(event)
        await self._send_snapshot_update(snapshot_mutate_event)

    async def _send_snapshot_update(self, snapshot_mutate_event):
        self._snapshot.merge_event(snapshot_mutate_event)
        out_cloudevent = CloudEvent(
            {
                "type": identifiers.EVTYPE_EE_SNAPSHOT_UPDATE,
                "source": "/ert/ee/0",
                "id": self.event_index(),
            },
            snapshot_mutate_event.to_dict(),
        )
        out_msg = to_json(out_cloudevent).decode()
        if out_msg and self._users:
            await asyncio.wait([user.send(out_msg) for user in self._users])

    def _wsocket(self):
        loop = self._loop
        asyncio.set_event_loop(loop)

        done = self._done

        async def handle_client(websocket, path):
            try:
                print(f"Client {websocket.remote_address} connected")
                self._users.add(websocket)

                data = self._snapshot.to_dict()
                out_cloudevent = CloudEvent(
                    {
                        "type": identifiers.EVTYPE_EE_SNAPSHOT,
                        "source": "/ert/ee/0",
                        "id": self.event_index(),
                    },
                    data,
                )
                message = to_json(out_cloudevent).decode()
                await websocket.send(message)

                async for message in websocket:
                    client_event = from_json(message)
                    if client_event["type"] == identifiers.EVTYPE_EE_TERMINATE_REQUEST:
                        print(f"Client {websocket.remote_address} asked to terminate")
                        self._stop()
            except Exception as e:
                import traceback

                print(e, traceback.format_exc())
            finally:
                print("Serverside: Client disconnected")
                self._users.remove(websocket)

        async def handle_dispatch(websocket, path):
            print(path)
            # dispatch_id = int(path.split("/")[2])  # assuming format is /dispatch/<id>
            # event_data = ee_entity.create_evaluator_event(dispatch_id, None, "started")
            # message = json.dumps(event_data.to_dict())
            # await asyncio.wait([user.send(message) for user in USERS])
            try:
                async for msg in websocket:
                    print(f"dispatch got: {msg}")
                    if msg == "null":
                        return
                    event = from_json(msg)
                    await dispatch.handle_event(self, event)

            except Exception as e:
                import traceback

                print(e, traceback.format_exc())
                if self._users:
                    message = self.terminate_message()
                    if self._users:
                        await asyncio.wait([user.send(message) for user in self._users])
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
                message = self.terminate_message()
                if self._users:
                    await asyncio.wait([user.send(message) for user in self._users])
                print("send terminated")
            print("server exiting")

        server_future = loop.create_task(evaluator_server(done))
        loop.run_until_complete(server_future)

        print("Evaluator thread Done")

    def terminate_message(self):
        out_cloudevent = CloudEvent(
            {
                "type": identifiers.EVTYPE_EE_TERMINATED,
                "source": "/ert/ee/0",
                "id": self.event_index(),
            }
        )
        message = to_json(out_cloudevent).decode()
        return message

    def event_index(self):
        index = self._event_index
        self._event_index += 1
        return index

    def run(self):
        self._ws_thread.start()
        self._ensemble.evaluate(self._host, self._port)
        return ee_monitor.create(self._host, self._port)

    def _stop(self):
        if not self._done.done():
            self._done.set_result(None)

    def stop(self):
        self._loop.call_soon_threadsafe(self._stop)


def get_job_id(path):
    return path.split("/")[11]


def get_step_id(path):
    return path.split("/")[9]


def get_stage_id(path):
    return path.split("/")[7]


def get_real_id(path):
    return path.split("/")[5]


def get_ee_id(path):
    return path.split("/")[3]
