import asyncio
import logging
import pickle
import sys
import threading
import time
from contextlib import contextmanager
from http import HTTPStatus
from typing import Optional, Set

import cloudevents.exceptions
import cloudpickle
import websockets
from cloudevents.http import from_json, to_json
from cloudevents.http.event import CloudEvent
from websockets.exceptions import ConnectionClosedError
from websockets.legacy.server import WebSocketServerProtocol

import ert_shared.ensemble_evaluator.monitor as ee_monitor
from ert.ensemble_evaluator import identifiers
from ert.serialization import evaluator_marshaller, evaluator_unmarshaller
from ert_shared.ensemble_evaluator.dispatch import Batcher, Dispatcher

if sys.version_info < (3, 7):
    from async_generator import asynccontextmanager
else:
    from contextlib import asynccontextmanager

logger = logging.getLogger(__name__)

_MAX_UNSUCCESSFUL_CONNECTION_ATTEMPTS = 3


class EnsembleEvaluator:
    # pylint: disable=too-many-instance-attributes
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
        self._dispatchers_connected: Optional[asyncio.Queue] = None
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

    @property
    def config(self):
        return self._config

    @property
    def ensemble(self):
        return self._ensemble

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
        data: Optional[dict] = None,
        extra_attrs: Optional[dict] = None,
        data_marshaller=evaluator_marshaller,
    ):
        if isinstance(data, dict):
            data["iter"] = self._iter
        if extra_attrs is None:
            extra_attrs = {}

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
                    message, data_unmarshaller=evaluator_unmarshaller
                )
                logger.debug(f"got message from client: {client_event}")
                if client_event["type"] == identifiers.EVTYPE_EE_USER_CANCEL:
                    logger.debug(f"Client {websocket.remote_address} asked to cancel.")
                    self._signal_cancel()

                elif client_event["type"] == identifiers.EVTYPE_EE_USER_DONE:
                    logger.debug(f"Client {websocket.remote_address} signalled done.")
                    self._stop()

    @asynccontextmanager
    async def count_dispatcher(self):
        if self._dispatchers_connected is None:
            self._dispatchers_connected = asyncio.Queue()
        await self._dispatchers_connected.put(None)
        yield
        await self._dispatchers_connected.get()
        self._dispatchers_connected.task_done()

    async def handle_dispatch(self, websocket, path):
        # pylint: disable=not-async-context-manager
        # (false positive)
        async with self.count_dispatcher():
            async for msg in websocket:
                try:
                    event = from_json(msg, data_unmarshaller=evaluator_unmarshaller)
                except cloudevents.exceptions.DataUnmarshallerError:
                    event = from_json(msg, data_unmarshaller=pickle.loads)
                if self._get_ee_id(event["source"]) != self._ee_id:
                    logger.info(
                        f"Got event from evaluator {self._get_ee_id(event['source'])} "
                        f"with source {event['source']}, "
                        f"ignoring since I am {self._ee_id}"
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
        # pylint: disable=no-member
        # (false positive)
        async with websockets.serve(
            self.connection_handler,
            sock=self._config.get_socket(),
            ssl=self._config.get_server_ssl_context(),
            process_request=self.process_request,
            max_queue=None,
            max_size=2**26,
        ):
            await done
            logger.debug("Got done signal.")
            # Wait for dispatchers to disconnect
            try:
                if self._dispatchers_connected is not None:
                    await asyncio.wait_for(
                        self._dispatchers_connected.join(), timeout=20
                    )
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
        return ee_monitor.create(self._config.get_connection_info())

    def _stop(self):
        if not self._done.done():
            self._done.set_result(None)

    def stop(self):
        self._loop.call_soon_threadsafe(self._stop)
        self._ws_thread.join()

    def _signal_cancel(self):
        """
        This is just a wrapper around logic for whether to sgnal cancel via
        a cancellable ensemble or to use internal stop-mechanism directly

        I.e. if the ensemble can be cancelled, it is, otherwise cancel
        is signalled internally. In both cases the evaluator waits for
        the  cancel-message to arrive before it shuts down properly.
        """
        if self._ensemble.cancellable:
            logger.debug("Cancelling current ensemble")
            self._ensemble.cancel()
        else:
            logger.debug("Stopping current ensemble")
            self._stop()

    def run_and_get_successful_realizations(self) -> int:
        unsuccessful_connection_attempts = 0
        with self.run() as monitor:
            while True:
                try:
                    for _ in monitor.track():
                        unsuccessful_connection_attempts = 0

                    logger.debug("Waiting for evaluator shutdown")
                    break
                except ConnectionClosedError as e:
                    logger.debug(
                        "Connection closed unexpectedly in "
                        f"run_and_get_successful_realizations: {e}"
                    )
                    # The monitor went away...  what the h*** does that mean?
                    # TODO: terminate and break out?
                except ConnectionRefusedError as e:
                    unsuccessful_connection_attempts += 1
                    logger.debug(
                        f"run_and_get_successful_realizations caught {e}."
                        f"{unsuccessful_connection_attempts} unsuccessful attempts"
                    )
                    if (
                        unsuccessful_connection_attempts
                        < _MAX_UNSUCCESSFUL_CONNECTION_ATTEMPTS
                    ):
                        # Simple backoff-strategy...
                        sleep_time = 0.25 * 2**unsuccessful_connection_attempts
                        logger.debug(
                            f"Sleeping for {sleep_time} seconds before attempting to reconnect"  # noqa: E501
                        )
                        time.sleep(sleep_time)
                    else:
                        logger.debug("Max connection attempts reached")
                        self._signal_cancel()
                        break
                except BaseException:  # pylint: disable=broad-except
                    logger.exception("unexpected error: ")
                    # We really don't know what happened...  shut down and
                    # get out of here. The monitor has been stopped by context
                    self._signal_cancel()
                    break

            self._ws_thread.join()
            logger.debug("Evaluator is done")
            return self._ensemble.get_successful_realizations()

    @staticmethod
    def _get_ee_id(source) -> str:
        # the ee_id will be found at /ert/ee/ee_id/...
        return source.split("/")[3]
