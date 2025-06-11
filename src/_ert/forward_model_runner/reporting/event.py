from __future__ import annotations

import asyncio
import logging
import queue
import threading
from pathlib import Path
from typing import Final

from _ert import events
from _ert.events import (
    ForwardModelStepChecksum,
    ForwardModelStepFailure,
    ForwardModelStepRunning,
    ForwardModelStepStart,
    ForwardModelStepSuccess,
    event_to_json,
)
from _ert.forward_model_runner.client import Client, ClientConnectionError
from _ert.forward_model_runner.reporting.base import Reporter
from _ert.forward_model_runner.reporting.message import (
    _STEP_EXIT_FAILED_STRING,
    Checksum,
    Exited,
    Finish,
    Init,
    Running,
    Start,
)
from _ert.forward_model_runner.reporting.statemachine import StateMachine
from _ert.threading import ErtThread

logger = logging.getLogger(__name__)


class EventSentinel:
    pass


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
    ) -> None:
        self._evaluator_url = evaluator_url
        self._token = token

        self._statemachine = StateMachine()
        self._statemachine.add_handler((Init,), self._init_handler)
        self._statemachine.add_handler((Start, Running, Exited), self._step_handler)
        self._statemachine.add_handler((Checksum,), self._checksum_handler)
        self._statemachine.add_handler((Finish,), self._finished_handler)

        self._ens_id = None
        self._real_id = None
        self._event_queue: queue.Queue[events.Event | EventSentinel] = queue.Queue()
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

    def stop(self, exited_event: Exited | None = None):
        if exited_event:
            self._statemachine.transition(exited_event)
        self._event_queue.put(Event._sentinel)
        self._done.set()
        if self._event_publisher_thread.is_alive():
            self._event_publisher_thread.join()

    def _event_publisher(self):
        async def publisher():
            async with Client(
                url=self._evaluator_url,
                token=self._token,
                ack_timeout=self._ack_timeout,
            ) as client:
                event = None
                start_time = None
                while True:
                    try:
                        if self._done.is_set() and start_time is None:
                            start_time = asyncio.get_event_loop().time()
                        if event is None:
                            event = self._event_queue.get()
                            if event is self._sentinel:
                                break
                        if (
                            start_time
                            and (asyncio.get_event_loop().time() - start_time)
                            > self._finished_event_timeout
                        ):
                            break
                        await client.send(event_to_json(event), self._max_retries)
                        event = None
                    except asyncio.CancelledError:
                        return
                    except ClientConnectionError as exc:
                        logger.error(f"Failed to send event: {exc}")
                        raise exc

        try:
            asyncio.run(publisher())
        except ClientConnectionError as exc:
            raise ClientConnectionError("Couldn't connect to evaluator") from exc

    def report(self, msg):
        self._statemachine.transition(msg)

    def _dump_event(self, event: events.Event):
        logger.debug(f'Schedule "{type(event)}" for delivery')
        self._event_queue.put(event)

    def _init_handler(self, msg: Init):
        self._ens_id = str(msg.ens_id)
        self._real_id = str(msg.real_id)
        self._event_publisher_thread.start()

    def _step_handler(self, msg: Start | Running | Exited):
        assert msg.step
        step_name = msg.step.name()
        step_msg = {
            "ensemble": self._ens_id,
            "real": self._real_id,
            "fm_step": str(msg.step.index),
        }
        if isinstance(msg, Start):
            logger.debug(f"Step {step_name} was successfully started")
            event = ForwardModelStepStart(
                **step_msg,
                std_out=str(Path(msg.step.std_out).resolve()),
                std_err=str(Path(msg.step.std_err).resolve()),
            )
            self._dump_event(event)
            if not msg.success():
                logger.error(f"Step {step_name} FAILED to start")
                event = ForwardModelStepFailure(**step_msg, error_msg=msg.error_message)
                self._dump_event(event)

        elif isinstance(msg, Exited):
            if msg.success():
                logger.debug(f"Step {step_name} exited successfully")
                self._dump_event(ForwardModelStepSuccess(**step_msg))
            else:
                logger.error(
                    _STEP_EXIT_FAILED_STRING.format(
                        step_name=msg.step.name(),
                        exit_code=msg.exit_code,
                        error_message=msg.error_message,
                    )
                )
                event = ForwardModelStepFailure(
                    **step_msg, exit_code=msg.exit_code, error_msg=msg.error_message
                )
                self._dump_event(event)

        elif isinstance(msg, Running):
            logger.debug(f"{step_name} step is running")
            event = ForwardModelStepRunning(
                **step_msg,
                max_memory_usage=msg.memory_status.max_rss,
                current_memory_usage=msg.memory_status.rss,
                cpu_seconds=msg.memory_status.cpu_seconds,
            )
            self._dump_event(event)

    def _finished_handler(self, _):
        self.stop()

    def _checksum_handler(self, msg: Checksum):
        fm_checksum = ForwardModelStepChecksum(
            ensemble=self._ens_id,
            real=self._real_id,
            checksums={msg.run_path: msg.data},
        )
        self._dump_event(fm_checksum)
