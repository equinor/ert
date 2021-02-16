import asyncio
import logging
import threading
import time
from typing import List

import async_generator
import ert_shared.ensemble_evaluator.entity.identifiers as identifiers
import ert_shared.ensemble_evaluator.monitor as ee_monitor
import fastapi
import uvicorn
import websockets
from cloudevents.http import from_json, to_json
from cloudevents.http.event import CloudEvent
from ert_shared.ensemble_evaluator.dispatch import Dispatcher
from ert_shared.ensemble_evaluator.entity.snapshot import (
    PartialSnapshot,
    Snapshot,
    _ForwardModel,
    _Job,
    _Realization,
    _SnapshotDict,
    _Stage,
    _Step,
)

logger = logging.getLogger(__name__)


class ConnectionManager:
    def __init__(self) -> None:
        self._active_connections: List[fastapi.WebSocket] = []

    async def connect(self, websocket: fastapi.WebSocket) -> None:
        await websocket.accept()
        self._active_connections.append(websocket)

    def disconnect(self, websocket: fastapi.WebSocket) -> None:
        self._active_connections.remove(websocket)

    async def broadcast(self, message: str, ignore_disconnects=False) -> None:
        for connection in self._active_connections:
            try:
                await connection.send_bytes(message)
            except (
                websockets.exceptions.ConnectionClosedOK,
                fastapi.WebSocketDisconnect,
            ):
                if not ignore_disconnects:
                    raise

    async def await_all_disconnected(self) -> None:
        while self._active_connections:
            await asyncio.sleep(0.2)


@async_generator.asynccontextmanager
async def managed_connection(
    connection_manager: ConnectionManager, websocket: fastapi.WebSocket
):
    await connection_manager.connect(websocket)
    yield
    connection_manager.disconnect(websocket)


class Server(uvicorn.Server):
    def install_signal_handlers(self):
        pass

    def start_in_thread(self):
        self._thread = threading.Thread(target=self.run)
        self._thread.start()
        while not self.started:
            time.sleep(1e-3)

    def stop_thread(self):
        self.should_exit = True

    def join(self, timeout=3):
        start = time.time()
        while self._thread.is_alive() and time.time() - start < timeout:
            time.sleep(1e-2)
        self.force_exit = True
        self._thread.join()


class EnsembleEvaluatorApp:
    _dispatch = Dispatcher()

    def __init__(
        self,
        snapshot,
        config,
        ee_id=0,
        ensemble_cancel_callback=lambda: None,
        server_stop_callback=lambda: None,
    ):
        self._ensemble_cancel_callback = ensemble_cancel_callback
        self._server_stop_callback = server_stop_callback
        self._snapshot = snapshot
        self._ee_id = ee_id
        self._config = config

        self._client_connection_manager = ConnectionManager()
        self._dispatcher_connection_manager = ConnectionManager()

        self._next_event_index = 1

    @_dispatch.register_event_handler(identifiers.EVGROUP_FM_ALL)
    async def _fm_handler(self, event):
        snapshot_mutate_event = PartialSnapshot(self._snapshot).from_cloudevent(event)
        await self._send_snapshot_update(snapshot_mutate_event)

    @_dispatch.register_event_handler(identifiers.EVTYPE_ENSEMBLE_STOPPED)
    async def _ensemble_stopped_handler(self, event):
        if self._snapshot.get_status() != "Failure":
            snapshot_mutate_event = PartialSnapshot(self._snapshot).from_cloudevent(
                event
            )
            await self._send_snapshot_update(snapshot_mutate_event)

    @_dispatch.register_event_handler(identifiers.EVTYPE_ENSEMBLE_STARTED)
    async def _ensemble_started_handler(self, event):
        if self._snapshot.get_status() != "Failure":
            snapshot_mutate_event = PartialSnapshot(self._snapshot).from_cloudevent(
                event
            )
            await self._send_snapshot_update(snapshot_mutate_event)

    @_dispatch.register_event_handler(identifiers.EVTYPE_ENSEMBLE_CANCELLED)
    async def _ensemble_cancelled_handler(self, event):
        if self._snapshot.get_status() != "Failure":
            snapshot_mutate_event = PartialSnapshot(self._snapshot).from_cloudevent(
                event
            )
            await self._send_snapshot_update(snapshot_mutate_event)
            await self._stop()

    @_dispatch.register_event_handler(identifiers.EVTYPE_ENSEMBLE_FAILED)
    async def _ensemble_failed_handler(self, event):
        if self._snapshot.get_status() not in ["Stopped", "Cancelled"]:
            snapshot_mutate_event = PartialSnapshot(self._snapshot).from_cloudevent(
                event
            )
            await self._send_snapshot_update(snapshot_mutate_event)

    async def _send_snapshot_update(self, snapshot_mutate_event):
        self._snapshot.merge_event(snapshot_mutate_event)
        out_cloudevent = CloudEvent(
            {
                "type": identifiers.EVTYPE_EE_SNAPSHOT_UPDATE,
                "source": f"/ert/ee/{self._ee_id}",
                "id": self.event_index(),
            },
            snapshot_mutate_event.to_dict(),
        )
        import pprint

        pp = pprint.PrettyPrinter(indent=2)
        print()
        pp.pprint(snapshot_mutate_event.to_dict())
        print(
            [
                (real_id, real["status"])
                for real_id, real in self._snapshot.to_dict()["reals"].items()
            ]
        )
        pp.pprint(self._snapshot.to_dict())
        print()
        out_msg = to_json(out_cloudevent).decode()
        if out_msg:
            await self._client_connection_manager.broadcast(
                message=out_msg, ignore_disconnects=True
            )

    @staticmethod
    def create_snapshot_msg(ee_id, snapshot, event_index):
        data = snapshot.to_dict()
        out_cloudevent = CloudEvent(
            {
                "type": identifiers.EVTYPE_EE_SNAPSHOT,
                "source": f"/ert/ee/{ee_id}",
                "id": event_index,
            },
            data,
        )
        return to_json(out_cloudevent).decode()

    async def handle_client(self, websocket: fastapi.WebSocket):
        async with managed_connection(self._client_connection_manager, websocket):
            try:
                message = self.create_snapshot_msg(
                    self._ee_id, self._snapshot, self.event_index()
                )
                await websocket.send_bytes(message)

                async for message in websocket.iter_bytes():
                    client_event = from_json(message)
                    logger.debug(f"got message from client: {client_event}")
                    if client_event["type"] == identifiers.EVTYPE_EE_USER_CANCEL:
                        logger.debug(f"Client {websocket} asked to cancel.")
                        if not self._ensemble_cancel_callback():
                            await self._stop()

                    if client_event["type"] == identifiers.EVTYPE_EE_USER_DONE:
                        logger.debug(f"Client {websocket} signalled done.")
                        await self._stop()
            except (
                websockets.exceptions.ConnectionClosedOK,
                fastapi.WebSocketDisconnect,
            ):
                return

    async def handle_dispatch(self, websocket: fastapi.WebSocket):
        async with managed_connection(self._dispatcher_connection_manager, websocket):
            try:
                async for message in websocket.iter_bytes():
                    event = from_json(message)
                    await self._dispatch.handle_event(self, event)
                    if event["type"] in [
                        identifiers.EVTYPE_ENSEMBLE_STOPPED,
                        identifiers.EVTYPE_ENSEMBLE_FAILED,
                    ]:
                        return
            except (
                websockets.exceptions.ConnectionClosedOK,
                fastapi.WebSocketDisconnect,
            ):
                return

    async def handle_test_connection(self, websocket: fastapi.WebSocket):
        await websocket.accept()
        try:
            async for _ in websocket.iter_bytes():
                pass
        except (websockets.exceptions.ConnectionClosedOK, fastapi.WebSocketDisconnect):
            return

    def terminate_message(self):
        out_cloudevent = CloudEvent(
            {
                "type": identifiers.EVTYPE_EE_TERMINATED,
                "source": f"/ert/ee/{self._ee_id}",
                "id": self.event_index(),
            }
        )
        message = to_json(out_cloudevent).decode()
        return message

    def event_index(self):
        index = self._next_event_index
        self._next_event_index += 1
        return index

    def get_successful_realizations(self):
        return self._snapshot.get_successful_realizations()

    async def _stop(self):
        try:
            await asyncio.wait_for(
                self._dispatcher_connection_manager.await_all_disconnected(),
                timeout=10,
            )
        except asyncio.TimeoutError:
            pass
        await self._client_connection_manager.broadcast(
            message=self.terminate_message(), ignore_disconnects=True
        )
        self._server_stop_callback()


class EnsembleEvaluator:
    def __init__(self, ensemble, config, ee_id=0):
        self._config = config
        self._ee_id = ee_id
        self._ensemble = ensemble
        self._ee_app = EnsembleEvaluatorApp(
            snapshot=self.create_snapshot(ensemble),
            config=config,
            ee_id=ee_id,
            ensemble_cancel_callback=self._cancel_ensemble,
            server_stop_callback=self._stop_server,
        )
        self._server = None

    @staticmethod
    def create_snapshot(ensemble):
        reals = {}
        for real in ensemble.get_active_reals():
            reals[str(real.get_iens())] = _Realization(
                active=True,
                start_time=None,
                end_time=None,
                status="Waiting",
            )
            for stage in real.get_stages():
                reals[str(real.get_iens())].stages[str(stage.get_id())] = _Stage(
                    status="Unknown",
                    start_time=None,
                    end_time=None,
                )
                for step in stage.get_steps():
                    reals[str(real.get_iens())].stages[str(stage.get_id())].steps[
                        str(step.get_id())
                    ] = _Step(status="Unknown", start_time=None, end_time=None)
                    for job in step.get_jobs():
                        reals[str(real.get_iens())].stages[str(stage.get_id())].steps[
                            str(step.get_id())
                        ].jobs[str(job.get_id())] = _Job(
                            status="Pending",
                            data={},
                            start_time=None,
                            end_time=None,
                            name=job.get_name(),
                        )
        top = _SnapshotDict(
            reals=reals,
            status="Unknown",
            forward_model=_ForwardModel(step_definitions={}),
            metadata=ensemble.get_metadata(),
        )

        return Snapshot(top.dict())

    def _cancel_ensemble(self):
        if self._ensemble.is_cancellable():
            # The evaluator will stop after the ensemble has
            # indicated it has been cancelled.
            self._ensemble.cancel()
            return True
        return False

    def run(self):
        app = fastapi.FastAPI()

        self._ee_app.handle_test_connection = app.websocket("/")(
            self._ee_app.handle_test_connection
        )
        self._ee_app.handle_client = app.websocket("/client")(
            self._ee_app.handle_client
        )
        self._ee_app.handle_dispatch = app.websocket("/dispatch")(
            self._ee_app.handle_dispatch
        )
        # TODO: Use socket instead
        config = uvicorn.Config(
            app,
            fd=self._config.get_socket().fileno(),
            log_level="info",
        )
        self._server = Server(config=config)
        self._server.start_in_thread()

        self._ensemble.evaluate(self._config, self._ee_id)
        return ee_monitor.create(self._config.host, self._config.port)

    def join(self):
        self._server.join()

    def _stop_server(self):
        self._server.stop_thread()

    def get_successful_realizations(self):
        return self._ee_app.get_successful_realizations()

    def run_and_get_successful_realizations(self):
        monitor = self.run()
        for _ in monitor.track():
            pass
        self._server.join()
        return self.get_successful_realizations()
