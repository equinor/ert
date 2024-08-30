import asyncio
import logging
import pickle
import traceback
from contextlib import asynccontextmanager, contextmanager
from http import HTTPStatus
from typing import (
    Any,
    AsyncIterator,
    Awaitable,
    Callable,
    Dict,
    Generator,
    Iterable,
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
from websockets.server import WebSocketServerProtocol

from ert.ensemble_evaluator import identifiers as ids
from ert.serialization import evaluator_marshaller, evaluator_unmarshaller

from ._ensemble import LegacyEnsemble as Ensemble
from .config import EvaluatorServerConfig
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
    EVTYPE_ENSEMBLE_SUCCEEDED,
    EVTYPE_FORWARD_MODEL_CHECKSUM,
)
from .snapshot import Snapshot
from .state import (
    ENSEMBLE_STATE_CANCELLED,
    ENSEMBLE_STATE_FAILED,
    ENSEMBLE_STATE_STOPPED,
)

logger = logging.getLogger(__name__)

EVENT_HANDLER = Callable[[List[CloudEvent]], Awaitable[None]]


class EnsembleEvaluator:
    def __init__(self, ensemble: Ensemble, config: EvaluatorServerConfig):
        self._config: EvaluatorServerConfig = config
        self._ensemble: Ensemble = ensemble

        self._loop: Optional[asyncio.AbstractEventLoop] = None

        self._clients: Set[WebSocketServerProtocol] = set()
        self._dispatchers_connected: asyncio.Queue[None] = asyncio.Queue()

        self._events: asyncio.Queue[CloudEvent] = asyncio.Queue()
        self._messages_to_send: asyncio.Queue[str] = asyncio.Queue()
        self._manifest_queue: asyncio.Queue[Any] = asyncio.Queue()

        self._result = None

        self._ee_tasks: List[asyncio.Task[None]] = []
        self._server_started: asyncio.Event = asyncio.Event()
        self._server_done: asyncio.Event = asyncio.Event()

        # batching section
        self._batch_processing_queue: asyncio.Queue[
            List[Tuple[EVENT_HANDLER, CloudEvent]]
        ] = asyncio.Queue()
        self._max_batch_size: int = 500
        self._batching_interval: int = 2

    async def _publisher(self) -> None:
        while True:
            msg = await self._messages_to_send.get()
            await asyncio.gather(
                *[client.send(msg) for client in self._clients],
                return_exceptions=True,
            )
            self._messages_to_send.task_done()

    async def _append_message(self, snapshot_update_event: Snapshot) -> None:
        message = self._create_cloud_message(
            EVTYPE_EE_SNAPSHOT_UPDATE,
            snapshot_update_event.to_dict(),
        )
        if message:
            await self._messages_to_send.put(message)

    async def _process_event_buffer(self) -> None:
        while True:
            batch = await self._batch_processing_queue.get()
            function_to_events_map: Dict[EVENT_HANDLER, List[CloudEvent]] = {}
            for func, event in batch:
                if func not in function_to_events_map:
                    function_to_events_map[func] = []
                function_to_events_map[func].append(event)

            for func, events in function_to_events_map.items():
                await func(events)

            self._batch_processing_queue.task_done()

    async def _batch_events_into_buffer(self) -> None:
        event_handler: Dict[str, EVENT_HANDLER] = {}

        def set_event_handler(event_types: Set[str], function: Any) -> None:
            for event_type in event_types:
                event_handler[event_type] = function

        set_event_handler(EVGROUP_FM_ALL, self._fm_handler)
        set_event_handler({EVTYPE_ENSEMBLE_STARTED}, self._started_handler)
        set_event_handler({EVTYPE_ENSEMBLE_SUCCEEDED}, self._stopped_handler)
        set_event_handler({EVTYPE_ENSEMBLE_CANCELLED}, self._cancelled_handler)
        set_event_handler({EVTYPE_ENSEMBLE_FAILED}, self._failed_handler)

        while True:
            batch: List[Tuple[EVENT_HANDLER, CloudEvent]] = []
            start_time = asyncio.get_running_loop().time()
            while (
                len(batch) < self._max_batch_size
                and asyncio.get_running_loop().time() - start_time
                < self._batching_interval
            ):
                try:
                    event = await asyncio.wait_for(self._events.get(), timeout=0.1)
                    function = event_handler[event["type"]]
                    batch.append((function, event))
                    self._events.task_done()
                except asyncio.TimeoutError:
                    continue
            await self._batch_processing_queue.put(batch)

    async def _fm_handler(self, events: List[CloudEvent]) -> None:
        await self._append_message(self.ensemble.update_snapshot(events))

    async def _started_handler(self, events: List[CloudEvent]) -> None:
        if self.ensemble.status != ENSEMBLE_STATE_FAILED:
            await self._append_message(self.ensemble.update_snapshot(events))

    async def _stopped_handler(self, events: List[CloudEvent]) -> None:
        if self.ensemble.status == ENSEMBLE_STATE_FAILED:
            return

        self._result = events[0].data  # normal termination
        max_memory_usage = -1
        for job in self.ensemble.snapshot.get_all_forward_models().values():
            memory_usage = job.get(ids.MAX_MEMORY_USAGE) or "-1"
            max_memory_usage = max(int(memory_usage), max_memory_usage)
        logger.info(
            f"Ensemble ran with maximum memory usage for a single realization job: {max_memory_usage}"
        )
        await self._append_message(self.ensemble.update_snapshot(events))

    async def _cancelled_handler(self, events: List[CloudEvent]) -> None:
        if self.ensemble.status != ENSEMBLE_STATE_FAILED:
            await self._append_message(self.ensemble.update_snapshot(events))
            self.stop()

    async def _failed_handler(self, events: List[CloudEvent]) -> None:
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
        await self._append_message(self.ensemble.update_snapshot(events))
        self._signal_cancel()  # let ensemble know it should stop

    @property
    def ensemble(self) -> Ensemble:
        return self._ensemble

    def _create_cloud_event(
        self,
        event_type: str,
        data: Optional[Dict[str, Any]] = None,
        extra_attrs: Optional[Dict[str, Any]] = None,
    ) -> CloudEvent:
        """Returns a CloudEvent with the given properties"""
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
        event = self._create_cloud_event(event_type, data, extra_attrs)
        return to_json(event, data_marshaller=data_marshaller).decode()

    @contextmanager
    def store_client(
        self, websocket: WebSocketServerProtocol
    ) -> Generator[None, None, None]:
        self._clients.add(websocket)
        yield
        self._clients.remove(websocket)

    async def handle_client(self, websocket: WebSocketServerProtocol) -> None:
        with self.store_client(websocket):
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
                    self.stop()

    @asynccontextmanager
    async def count_dispatcher(self) -> AsyncIterator[None]:
        await self._dispatchers_connected.put(None)
        yield
        await self._dispatchers_connected.get()
        self._dispatchers_connected.task_done()

    async def handle_dispatch(self, websocket: WebSocketServerProtocol) -> None:
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
                            await self._events.put(event)
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
                        EVTYPE_ENSEMBLE_SUCCEEDED,
                        EVTYPE_ENSEMBLE_FAILED,
                    ]:
                        return
            except ConnectionClosedError as connection_error:
                # Dispatchers may close the connection abruptly in the case of
                #  * flaky network (then the dispatcher will try to reconnect)
                #  * job being killed due to MAX_RUNTIME
                #  * job being killed by user
                logger.error(
                    f"a dispatcher abruptly closed a websocket: {str(connection_error)}"
                )

    async def forward_checksum(self, event: CloudEvent) -> None:
        forward_event = CloudEvent(
            {
                "type": EVTYPE_FORWARD_MODEL_CHECKSUM,
                "source": f"/ert/ensemble/{self.ensemble.id_}",
            },
            {event["run_path"]: event.data},
        )
        # clients still need to receive events via ws
        await self._messages_to_send.put(
            to_json(forward_event, data_marshaller=evaluator_marshaller).decode()
        )
        await self._manifest_queue.put(forward_event)

    async def connection_handler(
        self, websocket: WebSocketServerProtocol, path: str
    ) -> None:
        elements = path.split("/")
        if elements[1] == "client":
            await self.handle_client(websocket)
        elif elements[1] == "dispatch":
            await self.handle_dispatch(websocket)
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

    async def _server(self) -> None:
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
            self._server_started.set()
            await self._server_done.wait()
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
            await self._messages_to_send.put(message)
            await self._events.join()
            await self._batch_processing_queue.join()
            await self._messages_to_send.join()
        logger.debug("Async server exiting.")

    def stop(self) -> None:
        self._server_done.set()

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
            assert self._loop is not None
            self._loop.run_in_executor(None, self._ensemble.cancel)
        else:
            logger.debug("Stopping current ensemble")
            self.stop()

    async def _start_running(self) -> None:
        if not self._config:
            raise ValueError("no config for evaluator")
        self._loop = asyncio.get_running_loop()
        self._ee_tasks = [
            asyncio.create_task(self._server(), name="server_task"),
            asyncio.create_task(
                self._batch_events_into_buffer(), name="dispatcher_task"
            ),
            asyncio.create_task(self._process_event_buffer(), name="processing_task"),
            asyncio.create_task(self._publisher(), name="publisher_task"),
        ]
        # now we wait for the server to actually start
        await self._server_started.wait()

        self._ee_tasks.append(
            asyncio.create_task(
                self._ensemble.evaluate(
                    self._config, self._events, self._manifest_queue
                ),
                name="ensemble_task",
            )
        )

    async def _monitor_and_handle_tasks(self) -> None:
        pending: Iterable[asyncio.Task[None]] = self._ee_tasks

        while True:
            done, pending = await asyncio.wait(
                pending, return_when=asyncio.FIRST_COMPLETED
            )
            for task in done:
                if task_exception := task.exception():
                    exc_traceback = "".join(
                        traceback.format_exception(
                            None, task_exception, task_exception.__traceback__
                        )
                    )
                    logger.error(
                        (
                            f"Exception in evaluator task {task.get_name()}: {task_exception}\n"
                            f"Traceback: {exc_traceback}"
                        )
                    )
                    raise task_exception
                elif task.get_name() == "server_task":
                    return
                elif task.get_name() == "ensemble_task":
                    continue
                else:
                    msg = (
                        f"Something went wrong, {task.get_name()} is done prematurely!"
                    )
                    logger.error(msg)
                    raise RuntimeError(msg)

    async def run_and_get_successful_realizations(self) -> List[int]:
        await self._start_running()

        try:
            await self._monitor_and_handle_tasks()
        finally:
            for task in self._ee_tasks:
                if not task.done():
                    task.cancel()
            results = await asyncio.gather(*self._ee_tasks, return_exceptions=True)
            for result in results or []:
                if not isinstance(result, asyncio.CancelledError) and isinstance(
                    result, Exception
                ):
                    logger.error(str(result))
                    raise result
        logger.debug("Evaluator is done")
        return self._ensemble.get_successful_realizations()

    @staticmethod
    def _get_ens_id(source: str) -> str:
        # the ens_id will be found at /ert/ensemble/ens_id/...
        return source.split("/")[3]
