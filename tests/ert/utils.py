from __future__ import annotations

import asyncio
import contextlib
import time
import uuid
from collections import defaultdict
from collections.abc import Awaitable, Callable
from pathlib import Path
from types import ModuleType, TracebackType
from typing import TYPE_CHECKING, Any, Self

import zmq
import zmq.asyncio

from _ert.events import EnsembleEvaluationWarning
from _ert.forward_model_runner.client import (
    ACK_MSG,
    CONNECT_MSG,
    DISCONNECT_MSG,
    HEARTBEAT_MSG,
    TERMINATE_MSG,
)
from _ert.threading import ErtThread
from ert.scheduler.event import (
    FinishedEvent,
    StartedEvent,
)

if TYPE_CHECKING:
    from ert.scheduler.driver import Driver


import importlib.util
import sys
from collections.abc import Sequence
from copy import deepcopy
from datetime import datetime

from pydantic import BaseModel, Field

from ert.ensemble_evaluator.snapshot import (
    EnsembleSnapshot,
    EnsembleSnapshotMetadata,
    FMStepSnapshot,
    RealizationSnapshot,
    _filter_nones,
)


def import_from_location(name: str, location: str | None = None) -> ModuleType:
    spec = importlib.util.spec_from_file_location(name, location)
    if spec is None:
        raise ImportError(f"Could not find {name}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    if spec.loader is None:
        raise ImportError(f"No loader for {name}")
    spec.loader.exec_module(module)
    return module


class SnapshotBuilder(BaseModel):
    fm_steps: dict[str, FMStepSnapshot] = Field(default_factory=dict)
    metadata: Any = Field(
        default_factory=lambda: EnsembleSnapshotMetadata(
            fm_step_status=defaultdict(dict),
            real_status={},
            sorted_real_ids=[],
            sorted_fm_step_ids=defaultdict(list),
        )
    )

    def build(
        self,
        real_ids: Sequence[str],
        status: str | None,
        exec_hosts: str | None = None,
        start_time: datetime | None = None,
        end_time: datetime | None = None,
        message: str | None = None,
    ) -> EnsembleSnapshot:
        snapshot = EnsembleSnapshot()
        snapshot._ensemble_state = status
        snapshot._metadata = self.metadata

        for r_id in real_ids:
            snapshot.add_realization(
                r_id,
                RealizationSnapshot(
                    active=True,
                    fm_steps=deepcopy(self.fm_steps),
                    start_time=start_time,
                    end_time=end_time,
                    exec_hosts=exec_hosts,
                    status=status,
                    message=message,
                ),
            )
        return snapshot

    def add_fm_step(
        self,
        fm_step_id: str,
        index: str,
        name: str | None,
        status: str | None,
        current_memory_usage: int | None = None,
        max_memory_usage: int | None = None,
        start_time: datetime | None = None,
        end_time: datetime | None = None,
        stdout: str | None = None,
        stderr: str | None = None,
        error: str | None = None,
    ) -> SnapshotBuilder:
        self.fm_steps[fm_step_id] = _filter_nones(
            FMStepSnapshot(
                status=status,
                index=index,
                start_time=start_time,
                end_time=end_time,
                name=name,
                stdout=stdout,
                stderr=stderr,
                current_memory_usage=current_memory_usage,
                max_memory_usage=max_memory_usage,
                error=error,
            )
        )
        return self


def source_dir() -> Path:
    src = Path("@CMAKE_CURRENT_SOURCE_DIR@/../..")
    if src.is_dir():
        return src.relative_to(Path.cwd())

    # If the file was not correctly configured by cmake, look for the source
    # folder, assuming the build folder is inside the source folder.
    current_path = Path(__file__)
    while current_path != Path("/"):
        if (current_path / ".git").is_dir():
            return current_path
        # This is to find root dir for git worktrees
        if (current_path / ".git").is_file():
            with (current_path / ".git").open(encoding="utf-8") as f:
                for line in f:
                    if "gitdir:" in line:
                        return current_path

        current_path = current_path.parent
    raise RuntimeError("Cannot find the source folder")


SOURCE_DIR: Path = source_dir()


def wait_until(
    func: Callable[[], bool], interval: float = 0.5, timeout: float = 30.0
) -> None:
    """Waits until func returns True.

    Repeatedly calls 'func' until it returns true.
    Waits 'interval' seconds before each invocation. If 'timeout' is
    reached, will raise the AssertionError.
    """
    t = 0.0
    while t < timeout:
        time.sleep(interval)
        if func():
            return
        t += interval
    raise AssertionError(
        "Timeout reached in wait_until "
        f"(function {func.__name__}, timeout {timeout:g})."
    )


class MockZMQServer:
    def __init__(
        self,
        *,
        store_messages: bool = True,
        filtered_message_types: tuple[bytes, ...] = (CONNECT_MSG, DISCONNECT_MSG),
        no_response: bool = False,
        dont_ack_disconnect: bool = False,
        dont_ack_messages: bool = False,
    ) -> None:
        """Mock ZMQ server for testing

        store_messages: whether to store messages in self.messages
        filtered_message_types: Collection of message types that gets filtered out.
        no_response: Server will not respond to any messages.
        dont_ack_disconnect: Server will ack, but not to disconnect messages.
        dont_ack_messages: Server will not send ack to messages that are not connect
           or disconnect (effected by the dont_ack_disconnect parameter).
        """
        self.messages: list[str] = []
        self.loop: asyncio.AbstractEventLoop | None = None
        self.server_task: asyncio.Task[None] | None = None
        self.handler_task: asyncio.Task[None] | None = None
        self.dealers: set[bytes] = set()
        self.no_dealers = asyncio.Event()
        self.no_dealers.set()
        self.uri = f"ipc:///tmp/socket-{uuid.uuid4().hex[:8]}"

        self.filtered_message_types = filtered_message_types
        self.store_messages = store_messages
        self.no_response = no_response
        self.ack_disconnect = not dont_ack_disconnect
        self.ack_messages = not dont_ack_messages

    def start_event_loop(self) -> None:
        if self.loop is not None:
            asyncio.set_event_loop(self.loop)
            self.loop.run_until_complete(self.mock_zmq_server())

    def __enter__(self) -> Self:
        self.loop = asyncio.new_event_loop()
        self.thread = ErtThread(target=self.start_event_loop)
        self.thread.start()
        return self

    def __exit__(
        self,
        exc_type: object,
        exc_value: BaseException | None,
        traceback: TracebackType | None,
    ) -> None:
        if self.handler_task and not self.handler_task.done() and self.loop is not None:
            self.loop.call_soon_threadsafe(self.handler_task.cancel)
        self.thread.join()
        if self.loop is not None and not self.loop.is_closed():
            self.loop.close()

    async def __aenter__(self) -> Self:
        self.server_task = asyncio.create_task(self.mock_zmq_server())
        return self

    async def __aexit__(
        self,
        exc_type: object,
        exc_value: BaseException | None,
        traceback: TracebackType | None,
    ) -> None:
        if self.server_task is not None and not self.server_task.done():
            with contextlib.suppress(asyncio.TimeoutError):
                await asyncio.wait_for(self.no_dealers.wait(), timeout=2.0)
            self.server_task.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await self.server_task

    async def mock_zmq_server(self) -> None:
        zmq_context = zmq.asyncio.Context()
        self.router_socket = zmq_context.socket(zmq.ROUTER)
        self.router_socket.setsockopt(zmq.LINGER, 0)
        self.router_socket.bind(self.uri)

        self.handler_task = asyncio.create_task(self._handler())
        try:
            await self.handler_task
        finally:
            self.router_socket.close()
            zmq_context.term()

    async def do_heartbeat(self) -> None:
        for dealer in self.dealers:
            await self.router_socket.send_multipart([dealer, b"", HEARTBEAT_MSG])

    async def send_terminate_message(self) -> None:
        for dealer in self.dealers:
            await self.router_socket.send_multipart([dealer, b"", TERMINATE_MSG])

    async def _handler(self) -> None:
        while True:
            try:  # noqa: PLW0717
                dealer, __, frame = await self.router_socket.recv_multipart()
                if self.store_messages and frame not in self.filtered_message_types:
                    self.messages.append(frame.decode("utf-8"))
                if self.no_response:
                    continue
                if frame == CONNECT_MSG:
                    self.dealers.add(dealer)
                    self.no_dealers.clear()
                    await self.router_socket.send_multipart([dealer, b"", ACK_MSG])
                elif frame == DISCONNECT_MSG:
                    self.dealers.discard(dealer)
                    if not self.dealers:
                        self.no_dealers.set()
                    if self.ack_disconnect:
                        await self.router_socket.send_multipart([dealer, b"", ACK_MSG])
                elif self.ack_messages:
                    await self.router_socket.send_multipart([dealer, b"", ACK_MSG])
            except asyncio.CancelledError:
                break


async def poll(
    driver: Driver,
    expected: set[int],
    *,
    started: Callable[[list[int]], Awaitable[None]] | None = None,
    finished: Callable[[int, int], Awaitable[None]] | None = None,
    handle_warning: Callable[[EnsembleEvaluationWarning], None] | None = None,
) -> None:
    """Poll driver until expected realisations finish

    This function polls the given `driver` until realisations given by
    `expected` finish, either successfully or not, then returns. It is also
    possible to specify `started` and `finished` callbacks, for when a
    realisation starts and finishes, respectively. Blocks until all `expected`
    realisations finish.

    Parameters
    ----------
    driver : Driver
        Driver to poll
    expected : set[int]
        Set of realisation indices that we should wait for
    started : Callable[[list[int]], Awaitable[None]]
        Called for each job when it starts. A list containing the associated
        realisation index is passed.
    finished : Callable[[int, int], Awaitable[None]]
        Called for each job when it finishes. The first argument is the
        associated realisation index and the second is the returncode of the job
        process.
    handle_warning : Callable[[EnsembleEvaluationWarning], None]
        Called on EnsembleEvaluationWarnings from driver.

    """

    poll_task = asyncio.create_task(driver.poll())
    completed = set()
    try:
        while True:
            event = await driver.event_queue.get()
            if isinstance(event, StartedEvent):
                if started:
                    await started([event.iens])
            elif isinstance(event, FinishedEvent):
                if finished is not None:
                    await finished(event.iens, event.returncode)
                completed.add(event.iens)
                if completed == expected:
                    break
            elif (
                isinstance(event, EnsembleEvaluationWarning)
                and handle_warning is not None
            ):
                handle_warning(event)
    finally:
        poll_task.cancel()
        await driver.finish()
