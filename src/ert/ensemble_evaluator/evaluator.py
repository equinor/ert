from __future__ import annotations

import asyncio
import logging
import traceback
from collections.abc import Awaitable, Callable, Iterable, Sequence
from enum import Enum
from typing import Any, get_args

import zmq.asyncio

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
from _ert.forward_model_runner.client import (
    ACK_MSG,
    CONNECT_MSG,
    DISCONNECT_MSG,
    HEARTBEAT_MSG,
    HEARTBEAT_TIMEOUT,
)
from ert.ensemble_evaluator import identifiers as ids

from ..config import QueueSystem
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

EVENT_HANDLER = Callable[[list[Event]], Awaitable[None]]


class HeartbeatEvent(Enum):
    event = HEARTBEAT_MSG


class EnsembleEvaluator:
    def __init__(self, ensemble: Ensemble, config: EvaluatorServerConfig) -> None:
        self._config: EvaluatorServerConfig = config
        self._ensemble: Ensemble = ensemble

        self._events: asyncio.Queue[Event] = asyncio.Queue()
        self._events_to_send: asyncio.Queue[Event | HeartbeatEvent] = asyncio.Queue()
        self._manifest_queue: asyncio.Queue[Any] = asyncio.Queue()

        self._ee_tasks: list[asyncio.Task[None]] = []
        self._server_done: asyncio.Event = asyncio.Event()

        # batching section
        self._batch_processing_queue: asyncio.Queue[
            list[tuple[EVENT_HANDLER, Event]]
        ] = asyncio.Queue()
        self._max_batch_size: int = 500
        self._batching_interval: float = 0.5
        self._complete_batch: asyncio.Event = asyncio.Event()
        self._server_started: asyncio.Future[None] = asyncio.Future()
        self._clients_connected: set[bytes] = set()
        self._clients_empty: asyncio.Event = asyncio.Event()
        self._clients_empty.set()
        self._dispatchers_connected: set[bytes] = set()
        self._dispatchers_empty: asyncio.Event = asyncio.Event()
        self._dispatchers_empty.set()

    async def _do_heartbeat_clients(self) -> None:
        while True:
            if self._clients_connected:
                await self._events_to_send.put(HeartbeatEvent.event)
                await asyncio.sleep(HEARTBEAT_TIMEOUT)
            else:
                await asyncio.sleep(0.1)

    async def _publisher(self) -> None:
        while True:
            event = await self._events_to_send.get()
            for identity in self._clients_connected:
                if isinstance(event, HeartbeatEvent):
                    await self._router_socket.send_multipart(
                        [identity, b"", event.value]
                    )
                else:
                    await self._router_socket.send_multipart(
                        [identity, b"", event_to_json(event).encode("utf-8")]
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
            function_to_events_map: dict[EVENT_HANDLER, list[Event]] = {}
            for func, event in batch:
                if func not in function_to_events_map:
                    function_to_events_map[func] = []
                function_to_events_map[func].append(event)

            for func, events in function_to_events_map.items():
                await func(events)

            self._batch_processing_queue.task_done()

    async def _batch_events_into_buffer(self) -> None:
        event_handler: dict[type[Event], EVENT_HANDLER] = {}

        def set_event_handler(event_types: set[type[Event]], func: Any) -> None:
            for event_type in event_types:
                event_handler[event_type] = func

        set_event_handler(set(get_args(FMEvent | RealizationEvent)), self._fm_handler)
        set_event_handler({EnsembleStarted}, self._started_handler)
        set_event_handler({EnsembleSucceeded}, self._stopped_handler)
        set_event_handler({EnsembleCancelled}, self._cancelled_handler)
        set_event_handler({EnsembleFailed}, self._failed_handler)

        while True:
            batch: list[tuple[EVENT_HANDLER, Event]] = []
            start_time = asyncio.get_running_loop().time()
            while (
                len(batch) < self._max_batch_size
                and asyncio.get_running_loop().time() - start_time
                < self._batching_interval
            ):
                self._complete_batch.clear()
                try:
                    event = self._events.get_nowait()
                    function = event_handler[type(event)]
                    batch.append((function, event))
                    self._events.task_done()
                except asyncio.QueueEmpty:
                    await asyncio.sleep(0.1)
                    continue
            self._complete_batch.set()
            await self._batch_processing_queue.put(batch)
            if self._events.qsize() > 2 * self._max_batch_size:
                logger.info(f"{self._events.qsize()} events left in queue")

    async def _fm_handler(self, events: Sequence[FMEvent | RealizationEvent]) -> None:
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

            cpu_message = detect_overspent_cpu(
                self.ensemble.reals[int(real_id)].num_cpu, real_id, fm_step
            )
            if self.ensemble.queue_system != QueueSystem.LOCAL and cpu_message:
                logger.warning(cpu_message)

        logger.info(
            "Ensemble ran with maximum memory usage for a "
            f"single realization job: {max_memory_usage}"
        )

        await self._append_message(self.ensemble.update_snapshot(events))

    async def _cancelled_handler(self, events: Sequence[EnsembleCancelled]) -> None:
        if self.ensemble.status != ENSEMBLE_STATE_FAILED:
            await self._append_message(self.ensemble.update_snapshot(events))
            self.stop()

    async def _failed_handler(self, events: Sequence[EnsembleFailed]) -> None:
        if self.ensemble.status in {
            ENSEMBLE_STATE_STOPPED,
            ENSEMBLE_STATE_CANCELLED,
        }:
            return
        # if list is empty this call is not triggered by an
        # event, but as a consequence of some bad state
        # create a fake event because that's currently the only
        # api for setting state in the ensemble
        if len(events) == 0:
            events = [EnsembleFailed(ensemble=self.ensemble.id_)]
        await self._append_message(self.ensemble.update_snapshot(events))
        await self._signal_cancel()  # let ensemble know it should stop

    @property
    def ensemble(self) -> Ensemble:
        return self._ensemble

    async def handle_client(self, dealer: bytes, frame: bytes) -> None:
        if frame == CONNECT_MSG:
            if dealer in self._clients_connected:
                logger.warning(f"{dealer!r} wants to reconnect.")
            self._clients_connected.add(dealer)
            self._clients_empty.clear()
            current_snapshot_dict = self._ensemble.snapshot.to_dict()
            event: Event = EESnapshot(
                snapshot=current_snapshot_dict,
                ensemble=self.ensemble.id_,
            )
            await self._router_socket.send_multipart(
                [dealer, b"", event_to_json(event).encode("utf-8")]
            )
        elif frame == DISCONNECT_MSG:
            self._clients_connected.discard(dealer)
            if not self._clients_connected:
                self._clients_empty.set()
        else:
            event = event_from_json(frame.decode("utf-8"))
            if type(event) is EEUserCancel:
                logger.debug("Client asked to cancel.")
                await self._signal_cancel()
            elif type(event) is EEUserDone:
                logger.debug("Client signalled done.")
                self.stop()

    async def handle_dispatch(self, dealer: bytes, frame: bytes) -> None:
        if frame == CONNECT_MSG:
            self._dispatchers_connected.add(dealer)
            self._dispatchers_empty.clear()
        elif frame == DISCONNECT_MSG:
            self._dispatchers_connected.discard(dealer)
            if not self._dispatchers_connected:
                self._dispatchers_empty.set()
        else:
            event = dispatch_event_from_json(frame.decode("utf-8"))
            if event.ensemble != self.ensemble.id_:
                logger.info(
                    "Got event from evaluator "
                    f"{event.ensemble}. "
                    f"Ignoring since I am {self.ensemble.id_}"
                )
                return
            if type(event) is ForwardModelStepChecksum:
                await self.forward_checksum(event)
            else:
                await self._events.put(event)

    async def listen_for_messages(self) -> None:
        while True:
            try:
                dealer, _, frame = await self._router_socket.recv_multipart()
                await self._router_socket.send_multipart([dealer, b"", ACK_MSG])
                sender = dealer.decode("utf-8")
                if sender.startswith("client"):
                    await self.handle_client(dealer, frame)
                elif sender.startswith("dispatch"):
                    await self.handle_dispatch(dealer, frame)
                else:
                    logger.info(f"Connection attempt to unknown sender: {sender}.")
            except zmq.error.ZMQError as e:
                if e.errno == zmq.ENOTSOCK:
                    logger.warning(
                        "Evaluator receiver closed, no new messages are received"
                    )
                else:
                    logger.error(f"Unexpected error when listening to messages: {e}")
            except asyncio.CancelledError:
                return

    async def forward_checksum(self, event: Event) -> None:
        # clients still need to receive events via ws
        await self._events_to_send.put(event)
        await self._manifest_queue.put(event)

    async def _server(self) -> None:
        zmq_context = zmq.asyncio.Context()
        try:
            self._router_socket: zmq.asyncio.Socket = zmq_context.socket(zmq.ROUTER)
            self._router_socket.setsockopt(zmq.LINGER, 0)
            if self._config.server_public_key and self._config.server_secret_key:
                self._router_socket.curve_secretkey = self._config.server_secret_key
                self._router_socket.curve_publickey = self._config.server_public_key
                self._router_socket.curve_server = True

            if self._config.use_ipc_protocol:
                self._router_socket.bind(self._config.get_uri())
            else:
                self._config.router_port = self._router_socket.bind_to_random_port(
                    "tcp://*",
                    min_port=self._config.min_port,
                    max_port=self._config.max_port,
                )

            self._server_started.set_result(None)
        except zmq.error.ZMQBaseError as e:
            logger.error(f"ZMQ error encountered {e} during evaluator initialization")
            self._server_started.set_exception(e)
            zmq_context.destroy(linger=0)
            return
        try:
            await self._server_done.wait()
            try:
                await asyncio.wait_for(self._dispatchers_empty.wait(), timeout=5)
            except TimeoutError:
                logger.warning(
                    "Not all dispatchers were disconnected when closing zmq server!"
                )
            await self._events.join()
            await self._complete_batch.wait()
            await self._batch_processing_queue.join()
            event = EETerminated(ensemble=self._ensemble.id_)
            await self._events_to_send.put(event)
            await self._events_to_send.join()
            try:
                await asyncio.wait_for(self._clients_empty.wait(), timeout=5)
            except TimeoutError:
                logger.warning(
                    "Not all clients were disconnected when closing zmq server!"
                )
            logger.debug("Async server exiting.")
        finally:
            try:
                self._router_socket.close()
                zmq_context.destroy(linger=0)
            except Exception as exc:
                logger.warning(f"Failed to clean up zmq context {exc}")
            logger.info("ZMQ cleanup done!")

    def stop(self) -> None:
        self._server_done.set()

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
            self._ee_tasks.append(
                asyncio.create_task(
                    self._ensemble.cancel(), name="ensemble_cancellation_task"
                )
            )

        else:
            logger.debug("Stopping current ensemble")
            self.stop()

    async def _start_running(self) -> None:
        if not self._config:
            raise ValueError("no config for evaluator")
        self._ee_tasks = [asyncio.create_task(self._server(), name="server_task")]
        await self._server_started
        self._ee_tasks += [
            asyncio.create_task(self._do_heartbeat_clients(), name="heartbeat_task"),
            asyncio.create_task(
                self._batch_events_into_buffer(), name="dispatcher_task"
            ),
            asyncio.create_task(self._process_event_buffer(), name="processing_task"),
            asyncio.create_task(self._publisher(), name="publisher_task"),
            asyncio.create_task(self.listen_for_messages(), name="listener_task"),
            asyncio.create_task(
                self._ensemble.evaluate(
                    self._config, self._events, self._manifest_queue
                ),
                name="ensemble_task",
            ),
        ]

    CLOSE_SERVER_TIMEOUT = 60

    async def _monitor_and_handle_tasks(self) -> None:
        pending: Iterable[asyncio.Task[None]] = self._ee_tasks
        timeout = None

        while True:
            done, pending = await asyncio.wait(
                pending, return_when=asyncio.FIRST_COMPLETED, timeout=timeout
            )
            if not done and timeout:
                logger.info("Time-out while waiting for server_task to complete")
                self._server_done.set()
                timeout = None
                continue

            for task in done:
                if task_exception := task.exception():
                    self.log_exception(task_exception, task.get_name())
                    raise task_exception
                elif task.get_name() == "server_task":
                    return
                elif task.get_name() in {
                    "ensemble_task",
                    "listener_task",
                    "ensemble_cancellation_task",
                }:
                    timeout = self.CLOSE_SERVER_TIMEOUT
                else:
                    msg = (
                        f"Something went wrong, {task.get_name()} is done prematurely!"
                    )
                    logger.error(msg)
                    raise RuntimeError(msg)

    @staticmethod
    def log_exception(task_exception: BaseException, task_name: str) -> None:
        exc_traceback = "".join(
            traceback.format_exception(
                None, task_exception, task_exception.__traceback__
            )
        )
        logger.error(
            f"Exception in evaluator task {task_name}: {task_exception}\n"
            f"Traceback: {exc_traceback}"
        )

    async def run_and_get_successful_realizations(self) -> list[int]:
        await self._start_running()

        try:
            await self._monitor_and_handle_tasks()
        finally:
            self._server_done.set()
            self._clients_empty.set()
            self._dispatchers_empty.set()
            for task in self._ee_tasks:
                if not task.done():
                    task.cancel()
            results = await asyncio.gather(*self._ee_tasks, return_exceptions=True)
            for result in results or []:
                if not isinstance(result, asyncio.CancelledError) and isinstance(
                    result, Exception
                ):
                    logger.error(str(result))
                    raise RuntimeError(result) from result
        logger.debug("Evaluator is done")
        return self._ensemble.get_successful_realizations()

    @staticmethod
    def _get_ens_id(source: str) -> str:
        # the ens_id will be found at /ert/ensemble/ens_id/...
        return source.split("/")[3]


def detect_overspent_cpu(num_cpu: int, real_id: str, fm_step: FMStepSnapshot) -> str:
    """Produces a message warning about misconfiguration of NUM_CPU if
    so is detected. Returns an empty string if everything is ok."""
    allowed_overspending = 1.05
    minimum_wallclock_time_seconds = 30  # Information is only polled every 5 sec

    start_time = fm_step.get(ids.START_TIME)
    end_time = fm_step.get(ids.END_TIME)
    if start_time is None or end_time is None:
        return ""
    duration = (end_time - start_time).total_seconds()
    if duration <= minimum_wallclock_time_seconds:
        return ""
    cpu_seconds = fm_step.get(ids.CPU_SECONDS) or 0.0
    parallelization_obtained = cpu_seconds / duration
    if parallelization_obtained > num_cpu * allowed_overspending:
        return (
            f"Misconfigured NUM_CPU, forward model step '{fm_step.get(ids.NAME)}' for "
            f"realization {real_id} spent {cpu_seconds} cpu seconds "
            f"with wall clock duration {duration:.1f} seconds, "
            f"a factor of {parallelization_obtained:.2f}, while NUM_CPU was {num_cpu}."
        )
    return ""
