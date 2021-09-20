import asyncio
import logging
import threading
import sys
from contextlib import contextmanager
import pickle
from typing import Set
import cloudevents.exceptions
from http import HTTPStatus

import cloudpickle

import ert_shared.ensemble_evaluator.entity.identifiers as identifiers
import ert_shared.ensemble_evaluator.monitor as ee_monitor
import websockets
from websockets.legacy.server import WebSocketServerProtocol
from cloudevents.http import from_json, to_json
from cloudevents.http.event import CloudEvent
from ert_shared.ensemble_evaluator.dispatch import Dispatcher, Batcher
from ert_shared.ensemble_evaluator.entity import serialization
from ert_shared.status.entity.state import (
    ENSEMBLE_STATE_CANCELLED,
)

if sys.version_info < (3, 7):
    from async_generator import asynccontextmanager
else:
    from contextlib import asynccontextmanager

logger = logging.getLogger(__name__)


class EnsembleEvaluator:
    def __init__(self, ensemble, config, iter_, ee_id: str = "0"):
        # Without information on the iteration, the events emitted from the
        # evaluator are ambiguous. In the future, an experiment authority* will
        # "own" the evaluators and can add iteration information to events they
        # emit. In the mean time, it is added here.
        # * https://github.com/equinor/ert/issues/1250
        self._iter = iter_
        self._ee_id = ee_id
        self._config = config
        self._ensemble = ensemble

        self._loop = asyncio.new_event_loop()
        self._done = self._loop.create_future()

        self._clients: Set[WebSocketServerProtocol] = set()
        self._dispatchers_connected: asyncio.Queue[None] = asyncio.Queue(
            loop=self._loop
        )
        self._batcher = Batcher(timeout=2, loop=self._loop)
        self._dispatcher = Dispatcher(
            ensemble=self._ensemble,
            evaluator_callback=self.dispatcher_callback,
            batcher=self._batcher,
        )

        self._result = None

        self._ws_thread = threading.Thread(
            name="ert_ee_run_server", target=self._run_server, args=(self._loop,)
        )

    async def dispatcher_callback(self, event_type, snapshot_update_event, result=None):
        if event_type == identifiers.EVTYPE_ENSEMBLE_STOPPED:
            self._result = result

        await self._send_snapshot_update(snapshot_update_event)

        if event_type == identifiers.EVTYPE_ENSEMBLE_CANCELLED:
            self._stop()

    async def _send_snapshot_update(self, snapshot_update_event):
        event = self._create_cloud_event(
            identifiers.EVTYPE_EE_SNAPSHOT_UPDATE,
            snapshot_update_event.to_dict(),
        )
        if event and self._clients:
            await asyncio.gather(*[client.send(event) for client in self._clients])

    def _create_cloud_event(
        self,
        event_type,
        data={},
        extra_attrs={},
        data_marshaller=serialization.evaluator_marshaller,
    ):
        if isinstance(data, dict):
            data["iter"] = self._iter
        attrs = {
            "type": event_type,
            "source": f"/ert/ee/{self._ee_id}",
        }
        attrs.update(extra_attrs)
        out_cloudevent = CloudEvent(
            attrs,
            data,
        )
        return to_json(out_cloudevent, data_marshaller=data_marshaller).decode()

    @contextmanager
    def store_client(self, websocket):
        self._clients.add(websocket)
        yield
        self._clients.remove(websocket)

    async def handle_client(self, websocket, path):
        with self.store_client(websocket):
            event = self._create_cloud_event(
                identifiers.EVTYPE_EE_SNAPSHOT, self._ensemble.snapshot.to_dict()
            )
            await websocket.send(event)

            async for message in websocket:
                client_event = from_json(
                    message, data_unmarshaller=serialization.evaluator_unmarshaller
                )
                logger.debug(f"got message from client: {client_event}")
                if client_event["type"] == identifiers.EVTYPE_EE_USER_CANCEL:
                    logger.debug(f"Client {websocket.remote_address} asked to cancel.")
                    if self._ensemble.is_cancellable():
                        # The evaluator will stop after the ensemble has
                        # indicated it has been cancelled.
                        self._ensemble.cancel()
                    else:
                        self._stop()

                if client_event["type"] == identifiers.EVTYPE_EE_USER_DONE:
                    logger.debug(f"Client {websocket.remote_address} signalled done.")
                    self._stop()

    @asynccontextmanager
    async def count_dispatcher(self):
        await self._dispatchers_connected.put(None)
        yield
        await self._dispatchers_connected.get()
        self._dispatchers_connected.task_done()

    async def handle_dispatch(self, websocket, path):
        async with self.count_dispatcher():
            async for msg in websocket:
                try:
                    event = from_json(
                        msg, data_unmarshaller=serialization.evaluator_unmarshaller
                    )
                except cloudevents.exceptions.DataUnmarshallerError:
                    event = from_json(msg, data_unmarshaller=pickle.loads)
                if self._get_ee_id(event["source"]) != self._ee_id:
                    logger.info(
                        f"Got event from evaluator {self._get_ee_id(event['source'])} with source {event['source']}, ignoring since I am {self._ee_id}"
                    )
                    continue
                await self._dispatcher.handle_event(event)
                if event["type"] in [
                    identifiers.EVTYPE_ENSEMBLE_STOPPED,
                    identifiers.EVTYPE_ENSEMBLE_FAILED,
                ]:
                    return

    async def connection_handler(self, websocket, path):
        elements = path.split("/")
        if elements[1] == "client":
            await self.handle_client(websocket, path)
        elif elements[1] == "dispatch":
            await self.handle_dispatch(websocket, path)
        else:
            logger.info(f"Connection attempt to unknown path: {path}.")

    async def process_request(self, path, request_headers):
        if request_headers.get("token") != self._config.token:
            return HTTPStatus.UNAUTHORIZED, {}, b""
        if path == "/healthcheck":
            return HTTPStatus.OK, {}, b""

    async def evaluator_server(self, done):
        async with websockets.serve(
            self.connection_handler,
            sock=self._config.get_socket(),
            ssl=self._config.get_server_ssl_context(),
            process_request=self.process_request,
            max_queue=None,
            max_size=2 ** 26,
        ):
            await done
            logger.debug("Got done signal.")
            # Wait for dispatchers to disconnect
            try:
                await asyncio.wait_for(self._dispatchers_connected.join(), timeout=20)
            except asyncio.TimeoutError:
                logger.debug("Timed out waiting for dispatchers to disconnect")
            await self._batcher.join()

            terminated_attrs = {}
            terminated_data = None
            if self._result:
                terminated_attrs["datacontenttype"] = "application/octet-stream"
                terminated_data = cloudpickle.dumps(self._result)
            message = self._create_cloud_event(
                identifiers.EVTYPE_EE_TERMINATED,
                data=terminated_data,
                extra_attrs=terminated_attrs,
                data_marshaller=cloudpickle.dumps,
            )
            if self._clients:
                await asyncio.gather(
                    *[client.send(message) for client in self._clients]
                )
            logger.debug("Sent terminated to clients.")

        logger.debug("Async server exiting.")

    def _run_server(self, loop):
        loop.run_until_complete(self.evaluator_server(self._done))
        logger.debug("Server thread exiting.")

    def run(self) -> ee_monitor._Monitor:
        self._ws_thread.start()
        self._ensemble.evaluate(self._config, self._ee_id)
        return ee_monitor.create(
            self._config.host,
            self._config.port,
            self._config.protocol,
            self._config.cert,
            self._config.token,
        )

    def _stop(self):
        if not self._done.done():
            self._done.set_result(None)

    def stop(self):
        self._loop.call_soon_threadsafe(self._stop)
        self._ws_thread.join()

    def run_and_get_successful_realizations(self):
        try:
            with self.run() as mon:
                for _ in mon.track():
                    pass
        except ConnectionRefusedError as e:
            logger.debug(
                f"run_and_get_successful_realizations caught {e}, cancelling or stopping ensemble..."
            )
            if self._ensemble.is_cancellable():
                self._ensemble.cancel()
            else:
                self._stop()
            self._ws_thread.join()
            raise
        return self._ensemble.get_successful_realizations()

    @staticmethod
    def _get_ee_id(source) -> str:
        # the ee_id will be found at /ert/ee/ee_id/...
        return source.split("/")[3]
