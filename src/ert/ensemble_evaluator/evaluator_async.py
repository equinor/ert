import asyncio
import logging
import pickle
from collections import OrderedDict
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
from websockets.legacy.server import WebSocketServerProtocol

from ert.serialization import evaluator_marshaller, evaluator_unmarshaller

from ._builder import Ensemble
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
    EVTYPE_ENSEMBLE_STOPPED,
)
from .snapshot import PartialSnapshot
from .state import (
    ENSEMBLE_STATE_CANCELLED,
    ENSEMBLE_STATE_FAILED,
    ENSEMBLE_STATE_STOPPED,
)

logger = logging.getLogger(__name__)

_MAX_UNSUCCESSFUL_CONNECTION_ATTEMPTS = 3


class EnsembleEvaluatorAsync:
    def __init__(self, ensemble: Ensemble, config: EvaluatorServerConfig, iter_: int):
        # Without information on the iteration, the events emitted from the
        # evaluator are ambiguous. In the future, an experiment authority* will
        # "own" the evaluators and can add iteration information to events they
        # emit. In the meantime, it is added here.
        # * https://github.com/equinor/ert/issues/1250
        self._iter: int = iter_
        self._config: EvaluatorServerConfig = config
        self._ensemble: Ensemble = ensemble

        self._loop: Optional[asyncio.AbstractEventLoop] = None
        self._done: asyncio.Future[bool] = asyncio.Future()

        self._clients: Set[WebSocketServerProtocol] = set()
        self._dispatchers_connected: asyncio.Queue[None] = asyncio.Queue()

        self._events: asyncio.Queue[CloudEvent] = asyncio.Queue()
        self._messages: asyncio.Queue[str] = asyncio.Queue()

        self._result = None

        self._ee_tasks: List[asyncio.Task[None]] = []
        self._server_started: asyncio.Event = asyncio.Event()

        self._processing_queue: asyncio.Queue[
            List[Tuple[Callable[[List[CloudEvent]], Awaitable[None]], CloudEvent]]
        ] = asyncio.Queue()

    async def _publisher(self) -> None:
        try:
            while True:
                msg = await self._messages.get()
                await asyncio.gather(
                    *[client.send(msg) for client in self._clients],
                    return_exceptions=True,
                )
        except asyncio.CancelledError:
            # when cancelling the task, make sure to send the rest
            while not self._messages.empty():
                msg = self._messages.get_nowait()
                await asyncio.gather(
                    *[client.send(msg) for client in self._clients],
                    return_exceptions=True,
                )

    async def _append_message(self, snapshot_update_event: PartialSnapshot) -> None:
        message = await self._create_cloud_message(
            EVTYPE_EE_SNAPSHOT_UPDATE,
            snapshot_update_event.to_dict(),
        )
        if message:
            await self._messages.put(message)

    async def _process_buffer(self) -> None:
        while True:
            batch = await self._processing_queue.get()
            function_to_events_map: Dict[
                Callable[[List[CloudEvent]], Awaitable[None]], List[CloudEvent]
            ] = OrderedDict()
            for func, event in batch:
                if func not in function_to_events_map:
                    function_to_events_map[func] = []
                function_to_events_map[func].append(event)

            for func, events in function_to_events_map.items():
                await func(events)

            self._processing_queue.task_done()

    async def _dispatcher(self) -> None:
        event_handler = {}

        def set_handler(event_types: Set[str], function: Any) -> None:
            for event_type in event_types:
                event_handler[event_type] = function

        for e_type, f in (
            (EVGROUP_FM_ALL, self._fm_handler),
            ({EVTYPE_ENSEMBLE_STARTED}, self._started_handler),
            ({EVTYPE_ENSEMBLE_STOPPED}, self._stopped_handler),
            ({EVTYPE_ENSEMBLE_CANCELLED}, self._cancelled_handler),
            ({EVTYPE_ENSEMBLE_FAILED}, self._failed_handler),
        ):
            set_handler(e_type, f)

        while True:
            batch: List[
                Tuple[Callable[[List[CloudEvent]], Awaitable[None]], CloudEvent]
            ] = []
            start_time = asyncio.get_event_loop().time()
            while len(batch) < 500 and asyncio.get_event_loop().time() - start_time < 2:
                try:
                    event = await asyncio.wait_for(self._events.get(), timeout=0.1)
                    function = event_handler[event["type"]]
                    batch.append((function, event))
                    self._events.task_done()
                except asyncio.TimeoutError:
                    continue
            await self._processing_queue.put(batch)

    async def _fm_handler(self, events: List[CloudEvent]) -> None:
        await self._append_message(self.ensemble.update_snapshot(events))

    async def _started_handler(self, events: List[CloudEvent]) -> None:
        if self.ensemble.status != ENSEMBLE_STATE_FAILED:
            await self._append_message(self.ensemble.update_snapshot(events))

    async def _stopped_handler(self, events: List[CloudEvent]) -> None:
        if self.ensemble.status != ENSEMBLE_STATE_FAILED:
            self._result = events[0].data  # normal termination
            max_memory_usage = -1
            for job in self.ensemble.snapshot.get_all_forward_models().values():
                memory_usage = job.max_memory_usage or "-1"
                if int(memory_usage) > max_memory_usage:
                    max_memory_usage = int(memory_usage)
            logger.info(
                f"Ensemble ran with maximum memory usage for a single realization job: {max_memory_usage}"
            )
            await self._append_message(self.ensemble.update_snapshot(events))

    async def _cancelled_handler(self, events: List[CloudEvent]) -> None:
        if self.ensemble.status != ENSEMBLE_STATE_FAILED:
            await self._append_message(self.ensemble.update_snapshot(events))
            await self._stop()

    async def _failed_handler(self, events: List[CloudEvent]) -> None:
        if self.ensemble.status not in (
            ENSEMBLE_STATE_STOPPED,
            ENSEMBLE_STATE_CANCELLED,
        ):
            # if list is empty this call is not triggered by an
            # event, but as a consequence of some bad state
            # create a fake event because that's currently the only
            # api for setting state in the ensemble
            if len(events) == 0:
                events = [await self._create_cloud_event(EVTYPE_ENSEMBLE_FAILED)]
            await self._append_message(self.ensemble.update_snapshot(events))
            await self._signal_cancel()  # let ensemble know it should stop

    @property
    def config(self) -> EvaluatorServerConfig:
        return self._config

    @property
    def ensemble(self) -> Ensemble:
        return self._ensemble

    async def _create_cloud_event(
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

    async def _create_cloud_message(
        self,
        event_type: str,
        data: Optional[Dict[str, Any]] = None,
        extra_attrs: Optional[Dict[str, Any]] = None,
        data_marshaller: Optional[Callable[[Any], Any]] = evaluator_marshaller,
    ) -> str:
        """Creates the CloudEvent and returns the serialized json-string"""
        event = await self._create_cloud_event(event_type, data, extra_attrs)
        return to_json(event, data_marshaller=data_marshaller).decode()

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
            current_snapshot_dict = self._ensemble.snapshot.to_dict()
            event = await self._create_cloud_message(
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
                    await self._signal_cancel()

                elif client_event["type"] == EVTYPE_EE_USER_DONE:
                    logger.debug(f"Client {websocket.remote_address} signalled done.")
                    await self._stop()

    @asynccontextmanager
    async def count_dispatcher(self) -> AsyncIterator[None]:
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

            terminated_attrs: Dict[str, str] = {}
            terminated_data = None
            if self._result:
                terminated_attrs["datacontenttype"] = "application/octet-stream"
                terminated_data = cloudpickle.dumps(self._result)

            logger.debug("Sending termination-message to clients...")

            message = await self._create_cloud_message(
                EVTYPE_EE_TERMINATED,
                data=terminated_data,
                extra_attrs=terminated_attrs,
                data_marshaller=cloudpickle.dumps,
            )
            await self._messages.put(message)
        logger.debug("Async server exiting.")

    async def _stop(self) -> None:
        if not self._done.done():
            self._done.set_result(True)

    def stop(self) -> None:
        assert self._loop
        if self._loop.is_running():
            asyncio.run_coroutine_threadsafe(self._stop(), self._loop)

    async def _signal_cancel(self) -> None:
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
            await self._stop()

    async def _start_running(self) -> None:
        self._loop = asyncio.get_running_loop()
        self._ee_tasks = [
            asyncio.create_task(self._server(), name="server_task"),
            asyncio.create_task(self._dispatcher(), name="dispatcher_task"),
            asyncio.create_task(self._process_buffer(), name="processing_task"),
            asyncio.create_task(self._publisher(), name="publisher_task"),
            asyncio.create_task(
                self._ensemble.evaluate_async(self._config), name="ensemble_task"
            ),
        ]
        # now we wait for the server to actually start
        await self._server_started.wait()

    async def _monitor_and_handle_tasks(self) -> None:

        pending: Iterable[asyncio.Task[None]] = self._ee_tasks

        while True:
            done, pending = await asyncio.wait(
                pending, return_when=asyncio.FIRST_COMPLETED
            )
            for task in done:
                if task_exception := task.exception():
                    logger.error((f"Exception in evaluator task: {task_exception}"))
                    raise task_exception
                elif task.get_name() == "server_task":
                    return
                elif task.get_name() == "ensemble_task":
                    continue
                else:
                    logger.error(
                        f"Something went wrong, {task.get_name()} is done prematurely!"
                    )

    async def run_and_get_successful_realizations(self) -> List[int]:
        await self._start_running()

        try:
            await self._monitor_and_handle_tasks()
        finally:
            for task in self._ee_tasks:
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
