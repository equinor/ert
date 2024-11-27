from __future__ import annotations

import logging
import queue
import threading
from datetime import datetime, timedelta
from pathlib import Path
from typing import Final, Union

from _ert import events
from _ert.events import (
    ForwardModelStepChecksum,
    ForwardModelStepFailure,
    ForwardModelStepRunning,
    ForwardModelStepStart,
    ForwardModelStepSuccess,
    event_to_json,
)
from _ert.forward_model_runner.client import (
    Client,
    ClientConnectionClosedOK,
    ClientConnectionError,
)
from _ert.forward_model_runner.reporting.base import Reporter
from _ert.forward_model_runner.reporting.message import (
    _JOB_EXIT_FAILED_STRING,
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
    The Event reporter forwards events, coming from the running job, added with
    "report" to the given connection information.

    An Init event must be provided as the first message, which starts reporting,
    and a Finish event will signal the reporter that the last event has been reported.

    If event fails to be sent (e.g. due to connection error) it does not proceed to the
    next event but instead tries to re-send the same event.

    Whenever the Finish event (when all the jobs have exited) is provided
    the reporter will try to send all remaining events for a maximum of 60 seconds
    before stopping the reporter. Any remaining events will not be sent.
    """

    _sentinel: Final = EventSentinel()

    def __init__(self, evaluator_url, token=None, cert_path=None):
        self._evaluator_url = evaluator_url
        self._token = token
        if cert_path is not None:
            with open(cert_path, encoding="utf-8") as f:
                self._cert = f.read()
        else:
            self._cert = None

        self._statemachine = StateMachine()
        self._statemachine.add_handler((Init,), self._init_handler)
        self._statemachine.add_handler((Start, Running, Exited), self._job_handler)
        self._statemachine.add_handler((Checksum,), self._checksum_handler)
        self._statemachine.add_handler((Finish,), self._finished_handler)

        self._ens_id = None
        self._real_id = None
        self._event_queue: queue.Queue[events.Event | EventSentinel] = queue.Queue()
        self._event_publisher_thread = ErtThread(target=self._event_publisher)
        self._timeout_timestamp = None
        self._timestamp_lock = threading.Lock()
        # seconds to timeout the reporter the thread after Finish() was received
        self._reporter_timeout = 60

    def stop(self) -> None:
        self._event_queue.put(Event._sentinel)
        with self._timestamp_lock:
            self._timeout_timestamp = datetime.now() + timedelta(
                seconds=self._reporter_timeout
            )
        if self._event_publisher_thread.is_alive():
            self._event_publisher_thread.join()

    def _event_publisher(self):
        logger.debug("Publishing event.")
        with Client(
            url=self._evaluator_url,
            token=self._token,
            cert=self._cert,
        ) as client:
            event = None
            while True:
                with self._timestamp_lock:
                    if (
                        self._timeout_timestamp is not None
                        and datetime.now() > self._timeout_timestamp
                    ):
                        self._timeout_timestamp = None
                        break
                if event is None:
                    # if we successfully sent the event we can proceed
                    # to next one
                    event = self._event_queue.get()
                    if event is self._sentinel:
                        break
                try:
                    client.send(event_to_json(event))
                    event = None
                except ClientConnectionError as exception:
                    # Possible intermittent failure, we retry sending the event
                    logger.error(str(exception))
                except ClientConnectionClosedOK as exception:
                    # The receiving end has closed the connection, we stop
                    # sending events
                    logger.debug(str(exception))
                    break

    def report(self, msg):
        self._statemachine.transition(msg)

    def _dump_event(self, event: events.Event):
        logger.debug(f'Schedule "{type(event)}" for delivery')
        self._event_queue.put(event)

    def _init_handler(self, msg: Init):
        self._ens_id = str(msg.ens_id)
        self._real_id = str(msg.real_id)
        self._event_publisher_thread.start()

    def _job_handler(self, msg: Union[Start, Running, Exited]):
        assert msg.job
        job_name = msg.job.name()
        job_msg = {
            "ensemble": self._ens_id,
            "real": self._real_id,
            "fm_step": str(msg.job.index),
        }
        if isinstance(msg, Start):
            logger.debug(f"Job {job_name} was successfully started")
            event = ForwardModelStepStart(
                **job_msg,
                std_out=str(Path(msg.job.std_out).resolve()),
                std_err=str(Path(msg.job.std_err).resolve()),
            )
            self._dump_event(event)
            if not msg.success():
                logger.error(f"Job {job_name} FAILED to start")
                event = ForwardModelStepFailure(**job_msg, error_msg=msg.error_message)
                self._dump_event(event)

        elif isinstance(msg, Exited):
            if msg.success():
                logger.debug(f"Job {job_name} exited successfully")
                self._dump_event(ForwardModelStepSuccess(**job_msg))
            else:
                logger.error(
                    _JOB_EXIT_FAILED_STRING.format(
                        job_name=msg.job.name(),
                        exit_code=msg.exit_code,
                        error_message=msg.error_message,
                    )
                )
                event = ForwardModelStepFailure(
                    **job_msg, exit_code=msg.exit_code, error_msg=msg.error_message
                )
                self._dump_event(event)

        elif isinstance(msg, Running):
            logger.debug(f"{job_name} job is running")
            event = ForwardModelStepRunning(
                **job_msg,
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
