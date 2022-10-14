import asyncio
import logging
import pickle
import threading
import time
from contextlib import asynccontextmanager, contextmanager
from http import HTTPStatus
from typing import Optional, Set

import cloudevents.exceptions
import cloudpickle
import websockets
from aiohttp import ClientError
from cloudevents.conversion import to_json
from cloudevents.http import CloudEvent, from_json
from websockets.exceptions import ConnectionClosedError
from websockets.legacy.server import WebSocketServerProtocol

from ert.serialization import evaluator_marshaller, evaluator_unmarshaller

from ._builder import Ensemble
from .config import EvaluatorServerConfig
from .dispatch import BatchingDispatcher
from .identifiers import (
    EVGROUP_FM_ALL,
    EVTYPE_EE_SNAPSHOT,
    EVTYPE_EE_SNAPSHOT_UPDATE,
    EVTYPE_EE_TERMINATED,
    EVTYPE_EE_USER_CANCEL,
    EVTYPE_EE_USER_DONE,
    EVTYPE_ENSEMBLE_CANCELLED,
    EVTYPE_ENSEMBLE_FAILED,
    EVTYPE_ENSEMBLE_STARTED,
    EVTYPE_ENSEMBLE_STOPPED,
)
from .monitor import Monitor
from .state import (
    ENSEMBLE_STATE_CANCELLED,
    ENSEMBLE_STATE_FAILED,
    ENSEMBLE_STATE_STOPPED,
)

logger = logging.getLogger(__name__)

_MAX_UNSUCCESSFUL_CONNECTION_ATTEMPTS = 3


class EnsembleEvaluator:
    # pylint: disable=too-many-instance-attributes
    def __init__(self, ensemble: Ensemble, config: EvaluatorServerConfig, iter_: int):
        # Without information on the iteration, the events emitted from the
        # evaluator are ambiguous. In the future, an experiment authority* will
        # "own" the evaluators and can add iteration information to events they
        # emit. In the mean time, it is added here.
        # * https://github.com/equinor/ert/issues/1250
        self._iter: int = iter_
        self._config: EvaluatorServerConfig = config
        self._ensemble: Ensemble = ensemble

        self._loop = asyncio.new_event_loop()
        self._done = self._loop.create_future()

        self._clients: Set[WebSocketServerProtocol] = set()
        self._dispatchers_connected: asyncio.Queue = None
        self._dispatcher = BatchingDispatcher(
            self._loop,
            timeout=2,
            max_batch=1000,
        )

        for e_type, f in (
            (EVGROUP_FM_ALL, self._fm_handler),
            (EVTYPE_ENSEMBLE_STARTED, self._started_handler),
            (EVTYPE_ENSEMBLE_STOPPED, self._stopped_handler),
            (EVTYPE_ENSEMBLE_CANCELLED, self._cancelled_handler),
            (EVTYPE_ENSEMBLE_FAILED, self._failed_handler),
        ):
            self._dispatcher.register_event_handler(e_type, f)

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

    async def _fm_handler(self, events):
        snapshot_update_event = self.ensemble.update_snapshot(events)
        await self._send_snapshot_update(snapshot_update_event)

    async def _started_handler(self, events):
        if self.ensemble.status != ENSEMBLE_STATE_FAILED:
            snapshot_update_event = self.ensemble.update_snapshot(events)
            await self._send_snapshot_update(snapshot_update_event)

    async def _stopped_handler(self, events):
        if self.ensemble.status != ENSEMBLE_STATE_FAILED:
            self._result = events[0].data  # normal termination
            snapshot_update_event = self.ensemble.update_snapshot(events)
            await self._send_snapshot_update(snapshot_update_event)

    async def _cancelled_handler(self, events):
        if self.ensemble.status != ENSEMBLE_STATE_FAILED:
            snapshot_update_event = self.ensemble.update_snapshot(events)
            await self._send_snapshot_update(snapshot_update_event)
            self._stop()

    async def _failed_handler(self, events):
        if self.ensemble.status not in (
            ENSEMBLE_STATE_STOPPED,
            ENSEMBLE_STATE_CANCELLED,
        ):
            # if list is empty this call is not triggered by an
            # event, but as a consequence of some bad state
            # create a fake event because that's currently the only
            # api for setting state in the ensemble
            if len(events) == 0:
                events = [self._create_cloud_event(EVTYPE_ENSEMBLE_FAILED)]
            snapshot_update_event = self.ensemble.update_snapshot(events)
            await self._send_snapshot_update(snapshot_update_event)
            self._signal_cancel()  # let ensemble know it should stop

    async def _send_snapshot_update(self, snapshot_update_event):
        message = self._create_cloud_message(
            EVTYPE_EE_SNAPSHOT_UPDATE,
            snapshot_update_event.to_dict(),
        )
        if message and self._clients:
            # Note return_exceptions=True in gather. This fire-and-forget
            # approach is currently how we deal with failures when trying
            # to send udates to clients. Rationale is that if sending to
            # the client fails, the websocket is down and we have no way
            # to re-establish it. Thus, it becomes the responsibility of
            # the client to re-connect if necessary, in which case the first
            # update it receives will be a full snapshot.
            await asyncio.gather(
                *[client.send(message) for client in self._clients],
                return_exceptions=True,
            )

    def _create_cloud_event(
        self,
        event_type,
        data: Optional[dict] = None,
        extra_attrs: Optional[dict] = None,
    ):
        """Returns a CloudEvent with the given properties"""
        if isinstance(data, dict):
            data["iter"] = self._iter
        if extra_attrs is None:
            extra_attrs = {}

        attrs = {
            "type": event_type,
            "source": f"/ert/ensemble/{self.ensemble.id_}",
        }
        attrs.update(extra_attrs)
        return CloudEvent(
            attrs,
            data,
        )

    def _create_cloud_message(
        self,
        event_type,
        data: Optional[dict] = None,
        extra_attrs: Optional[dict] = None,
        data_marshaller=evaluator_marshaller,
    ):
        """Creates the CloudEvent and returns the serialized json-string"""
        return to_json(
            self._create_cloud_event(event_type, data, extra_attrs),
            data_marshaller=data_marshaller,
        ).decode()

    @contextmanager
    def store_client(self, websocket):
        self._clients.add(websocket)
        yield
        self._clients.remove(websocket)

    async def handle_client(self, websocket, path):
        with self.store_client(websocket):
            event = self._create_cloud_message(
                EVTYPE_EE_SNAPSHOT, self._ensemble.snapshot.to_dict()
            )
            await websocket.send(event)

            async for message in websocket:
                client_event = from_json(
                    message, data_unmarshaller=evaluator_unmarshaller
                )
                logger.debug(f"got message from client: {client_event}")
                if client_event["type"] == EVTYPE_EE_USER_CANCEL:
                    logger.debug(f"Client {websocket.remote_address} asked to cancel.")
                    self._signal_cancel()

                elif client_event["type"] == EVTYPE_EE_USER_DONE:
                    logger.debug(f"Client {websocket.remote_address} signalled done.")
                    self._stop()

    @asynccontextmanager
    async def count_dispatcher(self):
        # do this here (not in __init__) to ensure the queue
        # is created on the right event-loop
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
            try:
                async for msg in websocket:
                    try:
                        event = from_json(msg, data_unmarshaller=evaluator_unmarshaller)
                    except cloudevents.exceptions.DataUnmarshallerError:
                        event = from_json(msg, data_unmarshaller=pickle.loads)
                    if self._get_ens_id(event["source"]) != self.ensemble.id_:
                        logger.info(
                            "Got event from evaluator "
                            f"{self._get_ens_id(event['source'])} "
                            f"with source {event['source']}, "
                            f"ignoring since I am {self.ensemble.id_}"
                        )
                        continue
                    try:
                        await self._dispatcher.handle_event(event)
                    except BaseException as ex:  # pylint: disable=broad-except
                        # Exceptions include asyncio.InvalidStateError, and
                        # anything that self._*_handler() can raise (updates
                        # snapshots)
                        logger.warning(
                            "cannot handle event - "
                            f"closing connection to dispatcher: {ex}"
                        )
                        await websocket.close(
                            code=1011, reason=f"failed handling {event}"
                        )
                        return

                    if event["type"] in [
                        EVTYPE_ENSEMBLE_STOPPED,
                        EVTYPE_ENSEMBLE_FAILED,
                    ]:
                        return
            except ConnectionClosedError as connection_error:
                # Dispatchers my close the connection apruptly in the case of
                #  * flaky network (then the dispatcher will try to reconnect)
                #  * job being killed due to MAX_RUNTIME
                #  * job being killed by user
                logger.error(
                    f"a dispatcher abruptly closed a websocket: {str(connection_error)}"
                )

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

    async def evaluator_server(self):
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
            await self._done
            if self._dispatchers_connected is not None:
                logger.debug(
                    f"Got done signal. {self._dispatchers_connected.qsize()} "
                    "dispatchers to disconnect..."
                )
                try:  # Wait for dispatchers to disconnect
                    await asyncio.wait_for(
                        self._dispatchers_connected.join(), timeout=20
                    )
                except asyncio.TimeoutError:
                    logger.debug("Timed out waiting for dispatchers to disconnect")
            else:
                logger.debug("Got done signal. No dispatchers connected")

            logger.debug("Joining batcher...")
            try:
                await asyncio.wait_for(self._dispatcher.join(), timeout=20)
            except asyncio.TimeoutError:
                logger.debug("Timed out waiting for batcher")

            terminated_attrs = {}
            terminated_data = None
            if self._result:
                terminated_attrs["datacontenttype"] = "application/octet-stream"
                terminated_data = cloudpickle.dumps(self._result)

            logger.debug("Sending termination-message to clients...")
            message = self._create_cloud_message(
                EVTYPE_EE_TERMINATED,
                data=terminated_data,
                extra_attrs=terminated_attrs,
                data_marshaller=cloudpickle.dumps,
            )
            if self._clients:
                # See note about return_exceptions=True above
                await asyncio.gather(
                    *[client.send(message) for client in self._clients],
                    return_exceptions=True,
                )

        logger.debug("Async server exiting.")

    def _run_server(self, loop):
        loop.run_until_complete(self.evaluator_server())
        logger.debug("Server thread exiting.")

    def run(self) -> Monitor:
        self._ws_thread.start()
        self._ensemble.evaluate(self._config)
        return Monitor(self._config.get_connection_info())

    def _stop(self):
        if not self._done.done():
            self._done.set_result(None)

    def stop(self):
        self._loop.call_soon_threadsafe(self._stop)
        self._ws_thread.join()

    def _signal_cancel(self):
        """
        This is just a wrapper around logic for whether to signal cancel via
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
        monitor = self.run()
        unsuccessful_connection_attempts = 0
        while True:
            try:
                for _ in monitor.track():
                    unsuccessful_connection_attempts = 0
                break
            except (ConnectionClosedError) as e:
                logger.debug(
                    "Connection closed unexpectedly in "
                    f"run_and_get_successful_realizations: {e}"
                )
            except (ConnectionRefusedError, ClientError) as e:
                unsuccessful_connection_attempts += 1
                logger.debug(
                    f"run_and_get_successful_realizations caught {e}."
                    f"{unsuccessful_connection_attempts} unsuccessful attempts"
                )
                if (
                    unsuccessful_connection_attempts
                    == _MAX_UNSUCCESSFUL_CONNECTION_ATTEMPTS
                ):
                    logger.debug("Max connection attempts reached")
                    self._signal_cancel()
                    break

                sleep_time = 0.25 * 2**unsuccessful_connection_attempts
                logger.debug(
                    f"Sleeping for {sleep_time} seconds before attempting to reconnect"
                )
                time.sleep(sleep_time)
            except BaseException:  # pylint: disable=broad-except
                logger.exception("unexpected error: ")
                # We really don't know what happened...  shut down and
                # get out of here. Monitor is stopped by context-mgr
                self._signal_cancel()
                break

        logger.debug("Waiting for evaluator shutdown")
        self._ws_thread.join()
        logger.debug("Evaluator is done")
        return self._ensemble.get_successful_realizations()

    @staticmethod
    def _get_ens_id(source) -> str:
        # the ens_id will be found at /ert/ensemble/ens_id/...
        return source.split("/")[3]
