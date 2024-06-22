import asyncio
import logging
import pickle
import threading
from contextlib import asynccontextmanager, contextmanager
from http import HTTPStatus
from typing import (
    Any,
    AsyncIterator,
    Callable,
    Dict,
    Generator,
    List,
    Optional,
    Set,
    Tuple,
)

import cloudevents.exceptions
import cloudpickle
import websockets
from cloudevents.conversion import to_json
from cloudevents.http import CloudEvent, from_json
from websockets.datastructures import Headers, HeadersLike
from websockets.exceptions import ConnectionClosedError
from websockets.legacy.server import WebSocketServerProtocol

from _ert.async_utils import new_event_loop
from _ert.threading import ErtThread
from ert.ensemble_evaluator import identifiers as ids
from ert.serialization import evaluator_marshaller, evaluator_unmarshaller

from ._ensemble import LegacyEnsemble as Ensemble
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
    EVTYPE_FORWARD_MODEL_CHECKSUM,
)
from .snapshot import PartialSnapshot
from .state import (
    ENSEMBLE_STATE_CANCELLED,
    ENSEMBLE_STATE_FAILED,
    ENSEMBLE_STATE_STOPPED,
)

logger = logging.getLogger(__name__)


class EnsembleEvaluator:
    def __init__(self, ensemble: Ensemble, config: EvaluatorServerConfig, iter_: int):
        # Without information on the iteration, the events emitted from the
        # evaluator are ambiguous. In the future, an experiment authority* will
        # "own" the evaluators and can add iteration information to events they
        # emit. In the meantime, it is added here.
        # * https://github.com/equinor/ert/issues/1250
        self._iter: int = iter_
        self._config: EvaluatorServerConfig = config
        self._ensemble: Ensemble = ensemble

        self._loop = new_event_loop()
        self._done = self._loop.create_future()

        self._clients: Set[WebSocketServerProtocol] = set()
        self._dispatchers_connected: Optional[asyncio.Queue[None]] = None
        self._snapshot_mutex = threading.Lock()
        self._dispatcher = BatchingDispatcher(
            sleep_between_batches_seconds=2,
            max_batch=1000,
        )

        for e_type, f in (
            (EVGROUP_FM_ALL, self._fm_handler),
            ({EVTYPE_ENSEMBLE_STARTED}, self._started_handler),
            ({EVTYPE_ENSEMBLE_STOPPED}, self._stopped_handler),
            ({EVTYPE_ENSEMBLE_CANCELLED}, self._cancelled_handler),
            ({EVTYPE_ENSEMBLE_FAILED}, self._failed_handler),
        ):
            self._dispatcher.set_event_handler(e_type, f)

        self._result = None
        self._ws_thread = ErtThread(
            name="ert_ee_run_server", target=self._run_server, args=(self._loop,)
        )

    @property
    def config(self) -> EvaluatorServerConfig:
        return self._config

    @property
    def ensemble(self) -> Ensemble:
        return self._ensemble

    async def forward_checksum(self, event: CloudEvent) -> None:
        forward_event = CloudEvent(
            {
                "type": EVTYPE_FORWARD_MODEL_CHECKSUM,
                "source": f"/ert/ensemble/{self.ensemble.id_}",
            },
            {event["run_path"]: event.data},
        )
        await self._send_message(
            to_json(forward_event, data_marshaller=evaluator_marshaller).decode()
        )

    def _fm_handler(self, events: List[CloudEvent]) -> None:
        with self._snapshot_mutex:
            snapshot_update_event = self.ensemble.update_snapshot(events)
        send_future = asyncio.run_coroutine_threadsafe(
            self._send_snapshot_update(snapshot_update_event), self._loop
        )
        send_future.result()

    def _started_handler(self, events: List[CloudEvent]) -> None:
        if self.ensemble.status != ENSEMBLE_STATE_FAILED:
            with self._snapshot_mutex:
                snapshot_update_event = self.ensemble.update_snapshot(events)
            send_future = asyncio.run_coroutine_threadsafe(
                self._send_snapshot_update(snapshot_update_event), self._loop
            )
            send_future.result()

    def _stopped_handler(self, events: List[CloudEvent]) -> None:
        if self.ensemble.status == ENSEMBLE_STATE_FAILED:
            return

        self._result = events[0].data  # normal termination
        with self._snapshot_mutex:
            max_memory_usage = -1
            for job in self.ensemble.snapshot.get_all_forward_models().values():
                memory_usage = job.get(ids.MAX_MEMORY_USAGE) or "-1"
                max_memory_usage = max(int(memory_usage), max_memory_usage)
            logger.info(
                f"Ensemble ran with maximum memory usage for a single realization job: {max_memory_usage}"
            )
            snapshot_update_event = self.ensemble.update_snapshot(events)
        send_future = asyncio.run_coroutine_threadsafe(
            self._send_snapshot_update(snapshot_update_event), self._loop
        )
        send_future.result()

    def _cancelled_handler(self, events: List[CloudEvent]) -> None:
        if self.ensemble.status == ENSEMBLE_STATE_FAILED:
            return
        with self._snapshot_mutex:
            snapshot_update_event = self.ensemble.update_snapshot(events)
        send_future = asyncio.run_coroutine_threadsafe(
            self._send_snapshot_update(snapshot_update_event), self._loop
        )
        send_future.result()
        self._loop.call_soon_threadsafe(self._stop)

    def _failed_handler(self, events: List[CloudEvent]) -> None:
        if self.ensemble.status in (
            ENSEMBLE_STATE_STOPPED,
            ENSEMBLE_STATE_CANCELLED,
        ):
            return
        # if list is empty this call is not triggered by an
        # event, but as a consequence of some bad state
        # create a fake event because that's currently the only
        # api for setting state in the ensemble
        if len(events) == 0:
            events = [self._create_cloud_event(EVTYPE_ENSEMBLE_FAILED)]
        with self._snapshot_mutex:
            snapshot_update_event = self.ensemble.update_snapshot(events)
        send_future = asyncio.run_coroutine_threadsafe(
            self._send_snapshot_update(snapshot_update_event), self._loop
        )
        send_future.result()
        self._signal_cancel()  # let ensemble know it should stop

    async def _send_snapshot_update(
        self, snapshot_update_event: PartialSnapshot
    ) -> None:
        message = self._create_cloud_message(
            EVTYPE_EE_SNAPSHOT_UPDATE,
            snapshot_update_event.to_dict(),
        )
        await self._send_message(message)

    def _create_cloud_event(
        self,
        event_type: str,
        data: Optional[Dict[str, Any]] = None,
        extra_attrs: Optional[Dict[str, Any]] = None,
    ) -> CloudEvent:
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
        event_type: str,
        data: Optional[Dict[str, Any]] = None,
        extra_attrs: Optional[Dict[str, Any]] = None,
        data_marshaller: Optional[Callable[[Any], Any]] = evaluator_marshaller,
    ) -> str:
        """Creates the CloudEvent and returns the serialized json-string"""
        return to_json(
            self._create_cloud_event(event_type, data, extra_attrs),
            data_marshaller=data_marshaller,
        ).decode()

    @contextmanager
    def store_client(
        self, websocket: WebSocketServerProtocol
    ) -> Generator[None, None, None]:
        self._clients.add(websocket)
        yield
        self._clients.remove(websocket)

    async def handle_client(
        self, websocket: WebSocketServerProtocol, path: str
    ) -> None:
        with self.store_client(websocket):
            with self._snapshot_mutex:
                current_snapshot_dict = self._ensemble.snapshot.to_dict()
            event = self._create_cloud_message(
                EVTYPE_EE_SNAPSHOT, current_snapshot_dict
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
    async def count_dispatcher(self) -> AsyncIterator[None]:
        # do this here (not in __init__) to ensure the queue
        # is created on the right event-loop
        if self._dispatchers_connected is None:
            self._dispatchers_connected = asyncio.Queue()

        await self._dispatchers_connected.put(None)
        yield
        await self._dispatchers_connected.get()
        self._dispatchers_connected.task_done()

    async def handle_dispatch(
        self, websocket: WebSocketServerProtocol, path: str
    ) -> None:
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
                        if event["type"] == EVTYPE_FORWARD_MODEL_CHECKSUM:
                            await self.forward_checksum(event)
                        else:
                            await self._dispatcher.handle_event(event)
                    except BaseException as ex:
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

    async def connection_handler(
        self, websocket: WebSocketServerProtocol, path: str
    ) -> None:
        elements = path.split("/")
        if elements[1] == "client":
            await self.handle_client(websocket, path)
        elif elements[1] == "dispatch":
            await self.handle_dispatch(websocket, path)
        else:
            logger.info(f"Connection attempt to unknown path: {path}.")

    async def process_request(
        self, path: str, request_headers: Headers
    ) -> Optional[Tuple[HTTPStatus, HeadersLike, bytes]]:
        if request_headers.get("token") != self._config.token:
            return HTTPStatus.UNAUTHORIZED, {}, b""
        if path == "/healthcheck":
            return HTTPStatus.OK, {}, b""
        return None

    async def evaluator_server(self) -> None:
        async with websockets.serve(
            self.connection_handler,
            sock=self._config.get_socket(),
            ssl=self._config.get_server_ssl_context(),
            process_request=self.process_request,
            max_queue=None,
            max_size=2**26,
            ping_timeout=60,
            ping_interval=60,
            close_timeout=60,
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

            logger.debug("Waiting for batcher to finish...")
            try:
                await asyncio.wait_for(
                    self._dispatcher.wait_until_finished(), timeout=20
                )
            except asyncio.TimeoutError:
                logger.debug("Timed out waiting for batcher to finish")

            terminated_attrs: Dict[str, str] = {}
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

    def _run_server(self, loop: asyncio.AbstractEventLoop) -> None:
        loop.run_until_complete(self.evaluator_server())
        logger.debug("Server thread exiting.")

    def start_running(self) -> None:
        self._ws_thread.start()
        self._ensemble.evaluate(self._config)

    def _stop(self) -> None:
        if not self._done.done():
            self._done.set_result(None)

    def stop(self) -> None:
        self._loop.call_soon_threadsafe(self._stop)
        self._ws_thread.join()

    def _signal_cancel(self) -> None:
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
            self._loop.call_soon_threadsafe(self._stop)

    def join(self) -> None:
        self._ws_thread.join()
        logger.debug("Evaluator is done")

    def get_successful_realizations(self) -> List[int]:
        return self._ensemble.get_successful_realizations()

    @staticmethod
    def _get_ens_id(source: str) -> str:
        # the ens_id will be found at /ert/ensemble/ens_id/...
        return source.split("/")[3]

    async def _send_message(self, message: Optional[str] = None) -> None:
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
