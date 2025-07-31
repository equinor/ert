from __future__ import annotations

import asyncio
import logging
import os
import queue
import signal
import threading
import uuid
from pathlib import Path
from typing import Final, TypedDict

from _ert.events import (
    DispatcherEvent,
    ForwardModelStepChecksum,
    ForwardModelStepFailure,
    ForwardModelStepRunning,
    ForwardModelStepStart,
    ForwardModelStepSuccess,
    dispatcher_event_to_json,
)
from _ert.forward_model_runner.client import (
    Client,
    ClientConnectionError,
)
from _ert.forward_model_runner.reporting.base import Reporter
from _ert.forward_model_runner.reporting.message import (
    STEP_EXIT_FAILED_STRING_TEMPLATE,
    Checksum,
    Exited,
    Finish,
    Init,
    Message,
    Running,
    Start,
)
from _ert.forward_model_runner.reporting.statemachine import StateMachine
from _ert.threading import ErtThread

logger = logging.getLogger(__name__)


class EventSentinel:
    pass


class StepMessage(TypedDict):
    ensemble: str
    real: str
    fm_step: str


class Event(Reporter):
    """
    The Event reporter forwards events, coming from the running step, added with
    "report" to the given connection information.

    An Init event must be provided as the first message, which starts reporting,
    and a Finish event will signal the reporter that the last event has been reported.

    If event fails to be sent (e.g. due to connection error) it does not proceed to the
    next event but instead tries to re-send the same event.

    Whenever the Finish event (when all the steps have exited) is provided
    the reporter will try to send all remaining events for a maximum of 60 seconds
    before stopping the reporter. Any remaining events will not be sent.
    """

    _sentinel: Final = EventSentinel()

    def __init__(
        self,
        evaluator_url: str,
        token: str | None = None,
        ack_timeout: float | None = None,
        max_retries: int | None = None,
        finished_event_timeout: float | None = None,
        ens_id: str | None = None,
        real_id: str | None = None,
    ) -> None:
        self._evaluator_url = evaluator_url
        self._token = token

        self._statemachine = StateMachine()
        self._statemachine.add_handler((Init,), self._init_handler)
        self._statemachine.add_handler((Start, Running, Exited), self._step_handler)
        self._statemachine.add_handler((Checksum,), self._checksum_handler)
        self._statemachine.add_handler((Finish,), self._finished_handler)

        self._ens_id = ens_id
        self._real_id = real_id
        self._event_queue: queue.Queue[DispatcherEvent | EventSentinel] = queue.Queue()
        self._event_publisher_thread = ErtThread(
            target=self._event_publisher, should_raise=False
        )
        self._done = threading.Event()
        self._ack_timeout = ack_timeout
        self._max_retries = max_retries
        if finished_event_timeout is not None:
            self._finished_event_timeout = finished_event_timeout
        else:
            # for the sake of heavy load when using LOCAL_DRIVER
            # we set the default timeout to 10 minutes since forward model
            # can be finished but not all the events were sent yet
            self._finished_event_timeout = 600

    def stop(self, exited_event: Exited | None = None) -> None:
        if exited_event:
            self._statemachine.transition(exited_event)
        self._event_queue.put(Event._sentinel)
        self._done.set()
        if self._event_publisher_thread.is_alive():
            self._event_publisher_thread.join()

    async def handle_publish(self, client: Client) -> None:
        event = None
        start_time = None
        while True:
            try:
                if self._done.is_set() and start_time is None:
                    start_time = asyncio.get_event_loop().time()
                if event is None:
                    event = self._event_queue.get(timeout=0.1)
                    if event is self._sentinel:
                        break
                if (
                    start_time
                    and (asyncio.get_event_loop().time() - start_time)
                    > self._finished_event_timeout
                ):
                    break
                assert isinstance(event, DispatcherEvent)
                await client.send(dispatcher_event_to_json(event), self._max_retries)
                event = None
            except asyncio.CancelledError:
                return
            except ClientConnectionError as exc:
                logger.error(f"Failed to send event: {exc}")
                return
            except queue.Empty:
                await asyncio.sleep(0)

    async def listen_for_terminate_message(self, client: Client) -> None:
        try:
            await client.received_terminate_message.wait()

            logger.info("Received a TERMINATE message. Terminating the forward model")
            pgid = os.getpgid(os.getpid())
            os.killpg(pgid, signal.SIGTERM)
        except asyncio.CancelledError:
            return

    def _event_publisher(self) -> None:
        async def publisher() -> None:
            async with Client(
                url=self._evaluator_url,
                token=self._token,
                ack_timeout=self._ack_timeout,
                dealer_name=f"dispatch-real-{self._real_id}-{uuid.uuid4().hex[:6]}",
            ) as client:
                publisher_task = asyncio.create_task(
                    self.handle_publish(client), name="publisher_task"
                )
                listener_task = asyncio.create_task(
                    self.listen_for_terminate_message(client),
                    name="terminate_message_listener_task",
                )
                await publisher_task
                if not listener_task.done():
                    listener_task.cancel()
                    await listener_task

        asyncio.run(publisher())

    def report(self, msg: Message) -> None:
        self._statemachine.transition(msg)

    def _dump_event(self, event: DispatcherEvent) -> None:
        logger.debug(f'Schedule "{type(event)}" for delivery')
        self._event_queue.put(event)

    def _init_handler(self, msg: Init) -> None:
        self._ens_id = msg.ens_id
        self._real_id = str(msg.real_id)
        self._event_publisher_thread.start()

    def _step_handler(self, msg: Start | Running | Exited) -> None:
        step_name = msg.step.name() if msg.step is not None else "Unknown"
        assert self._ens_id is not None
        assert self._real_id is not None
        step_msg: StepMessage = {
            "ensemble": self._ens_id,
            "real": self._real_id if self._real_id is not None else "Unknown",
            "fm_step": str(msg.step.index if msg.step is not None else 0),
        }
        if isinstance(msg, Start):
            logger.debug(f"Step {step_name} was successfully started")
            assert msg.step is not None
            self._dump_event(
                ForwardModelStepStart(
                    **step_msg,
                    std_out=str(Path(msg.step.std_out).resolve())
                    if msg.step.std_out is not None
                    else None,
                    std_err=str(Path(msg.step.std_err).resolve())
                    if msg.step.std_err is not None
                    else None,
                )
            )
            if not msg.success():
                logger.error(f"Step {step_name} FAILED to start")
                error_message = (
                    msg.error_message
                    if msg.error_message is not None
                    else "Unknown error"
                )
                event = ForwardModelStepFailure(**step_msg, error_msg=error_message)
                self._dump_event(event)

        elif isinstance(msg, Exited):
            if msg.success():
                logger.debug(f"Step {step_name} exited successfully")
                self._dump_event(ForwardModelStepSuccess(**step_msg))
            else:
                logger.error(
                    STEP_EXIT_FAILED_STRING_TEMPLATE.format(
                        step_name=step_name,
                        exit_code=msg.exit_code,
                        error_message=msg.error_message,
                    )
                )
                error_message = (
                    msg.error_message
                    if msg.error_message is not None
                    else "Unknown error"
                )
                self._dump_event(
                    ForwardModelStepFailure(
                        **step_msg, exit_code=msg.exit_code, error_msg=error_message
                    )
                )

        elif isinstance(msg, Running):
            logger.debug(f"{step_name} step is running")
            self._dump_event(
                ForwardModelStepRunning(
                    **step_msg,
                    max_memory_usage=msg.memory_status.max_rss,
                    current_memory_usage=msg.memory_status.rss,
                    cpu_seconds=msg.memory_status.cpu_seconds,
                )
            )

    def _finished_handler(self, _: Finish) -> None:
        self.stop()

    def _checksum_handler(self, msg: Checksum) -> None:
        assert self._real_id is not None
        fm_checksum = ForwardModelStepChecksum(
            ensemble=self._ens_id,
            real=self._real_id,
            checksums={msg.run_path: msg.data},
        )
        self._dump_event(fm_checksum)
