import asyncio
import threading
import logging
import ert_shared.ensemble_evaluator.dispatch as dispatch
import ert_shared.ensemble_evaluator.entity.identifiers as identifiers
import ert_shared.ensemble_evaluator.monitor as ee_monitor
import websockets
from cloudevents.http import from_json, to_json
from cloudevents.http.event import CloudEvent
from ert_shared.ensemble_evaluator.entity.snapshot import (
    SnapshotBuilder,
    PartialSnapshot,
    Snapshot,
)
from async_generator import asynccontextmanager
from contextlib import contextmanager


logger = logging.getLogger(__name__)


class EnsembleEvaluator:
    def __init__(self, ensemble, config, ee_id=0):
        self._ee_id = ee_id

        self._config = config
        self._ensemble = ensemble

        self._loop = asyncio.new_event_loop()
        self._ws_thread = threading.Thread(
            name="ert_ee_run_server", target=self._run_server, args=(self._loop,)
        )
        self._done = self._loop.create_future()

        self._clients = set()
        self._dispatchers_connected = asyncio.Queue(loop=self._loop)

        self._snapshot = self.create_snapshot(ensemble)
        self._event_index = 1

    @staticmethod
    def create_snapshot(ensemble):
        builder = SnapshotBuilder()
        real = ensemble.get_reals()[0]
        for stage in real.get_stages():
            builder.add_stage(str(stage.get_id()), stage.get_status())
            for step in stage.get_steps():
                builder.add_step(str(stage.get_id()), str(step.get_id()), "Unknown")
                for job in step.get_jobs():
                    builder.add_job(
                        str(stage.get_id()),
                        str(step.get_id()),
                        str(job.get_id()),
                        job.get_name(),
                        "Unknown",
                        {},
                    )
        for key, val in ensemble.get_metadata().items():
            builder.add_metadata(key, val)
        return builder.build(
            [str(real.get_iens()) for real in ensemble.get_reals()], "Unknown"
        )

    @dispatch.register_event_handler(identifiers.EVGROUP_FM_ALL)
    async def _fm_handler(self, event):
        snapshot_mutate_event = PartialSnapshot.from_cloudevent(event)
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
        out_msg = to_json(out_cloudevent).decode()
        if out_msg and self._clients:
            await asyncio.wait([client.send(out_msg) for client in self._clients])

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

    @contextmanager
    def store_client(self, websocket):
        self._clients.add(websocket)
        yield
        self._clients.remove(websocket)

    async def handle_client(self, websocket, path):
        logger.debug(f"Client {websocket.remote_address} connected.")

        with self.store_client(websocket):
            message = self.create_snapshot_msg(
                self._ee_id, self._snapshot, self.event_index()
            )
            await websocket.send(message)

            async for message in websocket:
                client_event = from_json(message)
                if client_event["type"] == identifiers.EVTYPE_EE_TERMINATE_REQUEST:
                    logger.debug(
                        f"Client {websocket.remote_address} asked to terminate."
                    )
                    self._stop()
            logger.debug(f"Client {websocket.remote_address} disconnected.")

    @asynccontextmanager
    async def count_dispatcher(self):
        await self._dispatchers_connected.put(None)
        yield
        await self._dispatchers_connected.get()
        self._dispatchers_connected.task_done()

    async def handle_dispatch(self, websocket, path):
        logger.debug(f"Dispatch {websocket.remote_address} connected.")

        async with self.count_dispatcher():
            async for msg in websocket:
                logger.debug(f"Dispatch got: {msg}.")
                if msg == "null":
                    return
                event = from_json(msg)
                await dispatch.handle_event(self, event)
            logger.debug(f"Dispatch {websocket.remote_address} disconnected.")

    async def connection_handler(self, websocket, path):
        logger.debug(f"Connection handler started for {websocket.remote_address}.")
        elements = path.split("/")
        if elements[1] == "client":
            await self.handle_client(websocket, path)
        elif elements[1] == "dispatch":
            await self.handle_dispatch(websocket, path)

    async def evaluator_server(self, done):
        async with websockets.serve(
            self.connection_handler,
            self._config.get("host"),
            self._config.get("port"),
            max_queue=500,
            max_size=2 ** 26,
        ):
            await done
            logger.debug("Got done signal.")
            # give NFS adaptors and Queue adaptors some time to read/send last events
            try:
                await asyncio.wait_for(self._dispatchers_connected.join(), timeout=10)
            except asyncio.TimeoutError:
                pass
            message = self.terminate_message()
            if self._clients:
                await asyncio.wait([client.send(message) for client in self._clients])
            logger.debug("Sent terminated to clients.")

        logger.debug("Async server exiting.")

    def _run_server(self, loop):
        asyncio.set_event_loop(loop)

        server_future = asyncio.get_event_loop().create_task(
            self.evaluator_server(self._done)
        )
        asyncio.get_event_loop().run_until_complete(server_future)

        logger.debug("Server thread exiting.")

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
        index = self._event_index
        self._event_index += 1
        return index

    def run(self):
        self._ws_thread.start()
        mon = ee_monitor.create(self._config.get("host"), self._config.get("port"))
        self._ensemble.evaluate(self._config, self._ee_id, mon)
        return mon

    def _stop(self):
        if not self._done.done():
            self._done.set_result(None)

    def stop(self):
        self._loop.call_soon_threadsafe(self._stop)
        self._ws_thread.join()

    def get_successful_realizations(self):
        mon = self.run()
        for _ in mon.track():
            pass
        successful_reals = 0
        for real in self._snapshot.to_dict()["reals"].values():
            for stage in real["stages"].values():
                # FIXME: assumes one stage per real
                if stage["status"] == "Finished":
                    successful_reals += 1

        return successful_reals
