import asyncio
import datetime
import logging
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
    Sequence,
    Set,
    Tuple,
    Type,
    Union,
    get_args,
)

import websockets
from pydantic_core._pydantic_core import ValidationError
from websockets.datastructures import Headers, HeadersLike
from websockets.exceptions import ConnectionClosedError
from websockets.server import WebSocketServerProtocol

from _ert.events import (
    EESnapshot,
    EESnapshotUpdate,
    EETerminated,
    EEUserCancel,
    EEUserDone,
    EnsembleCancelled,
    EnsembleFailed,
    EnsembleStarted,
    EnsembleSucceeded,
    Event,
    FMEvent,
    ForwardModelStepChecksum,
    RealizationEvent,
    dispatch_event_from_json,
    event_from_json,
    event_to_json,
)
from ert.ensemble_evaluator import identifiers as ids

from ._ensemble import FMStepSnapshot
from ._ensemble import LegacyEnsemble as Ensemble
from .config import EvaluatorServerConfig
from .snapshot import EnsembleSnapshot
from .state import (
    ENSEMBLE_STATE_CANCELLED,
    ENSEMBLE_STATE_FAILED,
    ENSEMBLE_STATE_STOPPED,
)

logger = logging.getLogger(__name__)

EVENT_HANDLER = Callable[[List[Event]], Awaitable[None]]


class EnsembleEvaluator:
    def __init__(self, ensemble: Ensemble, config: EvaluatorServerConfig):
        self._config: EvaluatorServerConfig = config
        self._ensemble: Ensemble = ensemble

        self._loop: Optional[asyncio.AbstractEventLoop] = None

        self._clients: Set[WebSocketServerProtocol] = set()
        self._dispatchers_connected: asyncio.Queue[None] = asyncio.Queue()

        self._events: asyncio.Queue[Event] = asyncio.Queue()
        self._events_to_send: asyncio.Queue[Event] = asyncio.Queue()
        self._manifest_queue: asyncio.Queue[Any] = asyncio.Queue()

        self._ee_tasks: List[asyncio.Task[None]] = []
        self._server_started: asyncio.Event = asyncio.Event()
        self._server_done: asyncio.Event = asyncio.Event()

        # batching section
        self._batch_processing_queue: asyncio.Queue[
            List[Tuple[EVENT_HANDLER, Event]]
        ] = asyncio.Queue()
        self._max_batch_size: int = 500
        self._batching_interval: int = 2

    async def _publisher(self) -> None:
        while True:
            event = await self._events_to_send.get()
            await asyncio.gather(
                *[client.send(event_to_json(event)) for client in self._clients],
                return_exceptions=True,
            )
            self._events_to_send.task_done()

    async def _append_message(self, snapshot_update_event: EnsembleSnapshot) -> None:
        event = EESnapshotUpdate(
            snapshot=snapshot_update_event.to_dict(), ensemble=self._ensemble.id_
        )
        await self._events_to_send.put(event)

    async def _process_event_buffer(self) -> None:
        while True:
            batch = await self._batch_processing_queue.get()
            function_to_events_map: Dict[EVENT_HANDLER, List[Event]] = {}
            for func, event in batch:
                if func not in function_to_events_map:
                    function_to_events_map[func] = []
                function_to_events_map[func].append(event)

            for func, events in function_to_events_map.items():
                await func(events)

            self._batch_processing_queue.task_done()

    async def _batch_events_into_buffer(self) -> None:
        event_handler: Dict[Type[Event], EVENT_HANDLER] = {}

        def set_event_handler(event_types: Set[Type[Event]], func: Any) -> None:
            for event_type in event_types:
                event_handler[event_type] = func

        set_event_handler(
            set(get_args(Union[FMEvent, RealizationEvent])), self._fm_handler
        )
        set_event_handler({EnsembleStarted}, self._started_handler)
        set_event_handler({EnsembleSucceeded}, self._stopped_handler)
        set_event_handler({EnsembleCancelled}, self._cancelled_handler)
        set_event_handler({EnsembleFailed}, self._failed_handler)

        while True:
            batch: List[Tuple[EVENT_HANDLER, Event]] = []
            start_time = asyncio.get_running_loop().time()
            while (
                len(batch) < self._max_batch_size
                and asyncio.get_running_loop().time() - start_time
                < self._batching_interval
            ):
                try:
                    event = await asyncio.wait_for(self._events.get(), timeout=0.1)
                    function = event_handler[type(event)]
                    batch.append((function, event))
                    self._events.task_done()
                except asyncio.TimeoutError:
                    continue
            await self._batch_processing_queue.put(batch)

    async def _fm_handler(
        self, events: Sequence[Union[FMEvent, RealizationEvent]]
    ) -> None:
        await self._append_message(self.ensemble.update_snapshot(events))

    async def _started_handler(self, events: Sequence[EnsembleStarted]) -> None:
        if self.ensemble.status != ENSEMBLE_STATE_FAILED:
            await self._append_message(self.ensemble.update_snapshot(events))

    async def _stopped_handler(self, events: Sequence[EnsembleSucceeded]) -> None:
        if self.ensemble.status == ENSEMBLE_STATE_FAILED:
            return

        max_memory_usage = -1
        for (real_id, _), fm_step in self.ensemble.snapshot.get_all_fm_steps().items():
            # Infer max memory usage
            memory_usage = fm_step.get(ids.MAX_MEMORY_USAGE) or "-1"
            max_memory_usage = max(int(memory_usage), max_memory_usage)

            if cpu_message := detect_overspent_cpu(
                self.ensemble.reals[int(real_id)].num_cpu, real_id, fm_step
            ):
                logger.warning(cpu_message)

        logger.info(
            f"Ensemble ran with maximum memory usage for a single realization job: {max_memory_usage}"
        )

        await self._append_message(self.ensemble.update_snapshot(events))

    async def _cancelled_handler(self, events: Sequence[EnsembleCancelled]) -> None:
        if self.ensemble.status != ENSEMBLE_STATE_FAILED:
            await self._append_message(self.ensemble.update_snapshot(events))
            self.stop()

    async def _failed_handler(self, events: Sequence[EnsembleFailed]) -> None:
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
            events = [EnsembleFailed(ensemble=self.ensemble.id_)]
        await self._append_message(self.ensemble.update_snapshot(events))
        self._signal_cancel()  # let ensemble know it should stop

    @property
    def ensemble(self) -> Ensemble:
        return self._ensemble

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
            event: Event = EESnapshot(
                snapshot=current_snapshot_dict, ensemble=self.ensemble.id_
            )
            await websocket.send(event_to_json(event))

            async for raw_msg in websocket:
                event = event_from_json(raw_msg)
                logger.debug(f"got message from client: {event}")
                if type(event) is EEUserCancel:
                    logger.debug(f"Client {websocket.remote_address} asked to cancel.")
                    self._signal_cancel()

                elif type(event) is EEUserDone:
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
                async for raw_msg in websocket:
                    try:
                        event = dispatch_event_from_json(raw_msg)
                        if event.ensemble != self.ensemble.id_:
                            logger.info(
                                "Got event from evaluator "
                                f"{event.ensemble}. "
                                f"Ignoring since I am {self.ensemble.id_}"
                            )
                            continue
                        if type(event) is ForwardModelStepChecksum:
                            await self.forward_checksum(event)
                        else:
                            await self._events.put(event)
                    except ValidationError as ex:
                        logger.warning(
                            "cannot handle event - "
                            f"closing connection to dispatcher: {ex}"
                        )
                        await websocket.close(
                            code=1011, reason=f"failed handling {event}"
                        )
                        return

                    if type(event) in [EnsembleSucceeded, EnsembleFailed]:
                        return
            except ConnectionClosedError as connection_error:
                # Dispatchers may close the connection abruptly in the case of
                #  * flaky network (then the dispatcher will try to reconnect)
                #  * job being killed due to MAX_RUNTIME
                #  * job being killed by user
                logger.error(
                    f"a dispatcher abruptly closed a websocket: {connection_error!s}"
                )

    async def forward_checksum(self, event: Event) -> None:
        # clients still need to receive events via ws
        await self._events_to_send.put(event)
        await self._manifest_queue.put(event)

    async def connection_handler(self, websocket: WebSocketServerProtocol) -> None:
        path = websocket.path
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

            logger.debug("Sending termination-message to clients...")

            event = EETerminated(ensemble=self._ensemble.id_)
            await self._events_to_send.put(event)
            await self._events.join()
            await self._batch_processing_queue.join()
            await self._events_to_send.join()
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


def detect_overspent_cpu(num_cpu: int, real_id: str, fm_step: FMStepSnapshot) -> str:
    """Produces a message warning about misconfiguration of NUM_CPU if
    so is detected. Returns an empty string if everything is ok."""
    now = datetime.datetime.now()
    duration = (
        (fm_step.get(ids.END_TIME) or now) - (fm_step.get(ids.START_TIME) or now)
    ).total_seconds()
    if duration <= 0:
        return ""
    cpu_seconds = fm_step.get(ids.CPU_SECONDS) or 0.0
    parallelization_obtained = cpu_seconds / duration
    if parallelization_obtained > num_cpu:
        return (
            f"Misconfigured NUM_CPU, forward model step '{fm_step.get(ids.NAME)}' for "
            f"realization {real_id} spent {cpu_seconds} cpu seconds "
            f"with wall clock duration {duration:.1f} seconds, "
            f"a factor of {parallelization_obtained:.2f}, while NUM_CPU was {num_cpu}."
        )
    return ""
