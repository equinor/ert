import datetime
import logging
import queue
import threading
from pathlib import Path
from typing import Any, Dict

from cloudevents.conversion import to_json
from cloudevents.http import CloudEvent

from _ert_job_runner.client import (
    Client,
    ClientConnectionClosedOK,
    ClientConnectionError,
)
from _ert_job_runner.reporting.base import Reporter
from _ert_job_runner.reporting.message import (
    _JOB_EXIT_FAILED_STRING,
    Exited,
    Finish,
    Init,
    Message,
    Running,
    Start,
)
from _ert_job_runner.reporting.statemachine import StateMachine

_FM_JOB_START = "com.equinor.ert.forward_model_job.start"
_FM_JOB_RUNNING = "com.equinor.ert.forward_model_job.running"
_FM_JOB_SUCCESS = "com.equinor.ert.forward_model_job.success"
_FM_JOB_FAILURE = "com.equinor.ert.forward_model_job.failure"

_CONTENT_TYPE = "datacontenttype"
_JOB_MSG_TYPE = "type"
_JOB_SOURCE = "source"

logger = logging.getLogger(__name__)


class Event(Reporter):
    """
    The Event reporter forwards events, coming from the running job, added with
    "report" to the given connection information.

    An Init event must provided as the first message, which starts reporting,
    and a Finish event will signal the reporter that the last event has been reported.

    If event fails to be sent (eg. due to connection error) it does not proceed to the
    next event but instead tries to re-send the same event.

    Whenever the Finish event (when all the jobs have exited) is provided
    the reporter will try to send all remaining events for a maximum of 60 seconds
    before stopping the reporter. Any remaining events will not be sent.
    """

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
        self._statemachine.add_handler((Finish,), self._finished_handler)

        self._ens_id = None
        self._real_id = None
        self._event_queue = queue.Queue()
        self._event_publisher_thread = threading.Thread(target=self._event_publisher)
        self._sentinel = object()  # notifying the queue's ended
        self._timeout_timestamp = None
        self._timestamp_lock = threading.Lock()
        # seconds to timeout the reporter the thread after Finish() was received
        self._reporter_timeout = 60

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
                        and datetime.datetime.now() > self._timeout_timestamp
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
                    client.send(to_json(event).decode())
                    event = None
                except ClientConnectionError as exception:
                    # Possible intermittent failure, we retry sending the event
                    logger.error(str(exception))
                    pass
                except ClientConnectionClosedOK as exception:
                    # The receiving end has closed the connection, we stop
                    # sending events
                    logger.debug(str(exception))
                    break

    def report(self, msg):
        self._statemachine.transition(msg)

    def _dump_event(self, attributes: Dict[str, str], data: Any = None):
        if data is None and _CONTENT_TYPE in attributes:
            attributes.pop(_CONTENT_TYPE)

        event = CloudEvent(attributes=attributes, data=data)
        logger.debug(f'Schedule {type(event)} "{event["type"]}" for delivery')
        self._event_queue.put(event)

    def _init_handler(self, msg):
        self._ens_id = msg.ens_id
        self._real_id = msg.real_id
        self._event_publisher_thread.start()

    def _job_msg_attrs(self, job):
        return {
            _JOB_SOURCE: (
                f"/ert/ensemble/{self._ens_id}/real/{self._real_id}/"
                f"job/{job.index}/index/{job.index}"
            ),
            _CONTENT_TYPE: "application/json",
        }

    def _job_handler(self, msg: Message):
        match msg:
            case Start(job, error_message=None):
                logger.debug(f"Job {job.name} was successfully started")
                self._dump_event(
                    {_JOB_MSG_TYPE: _FM_JOB_START, **self._job_msg_attrs(job)},
                    {
                        "stdout": str(Path(msg.job.std_out).resolve()),
                        "stderr": str(Path(msg.job.std_err).resolve()),
                    },
                )
            case Start(job, error_message=err):
                logger.error(f"Job {job.name} FAILED to start")
                self._dump_event(
                    {_JOB_MSG_TYPE: _FM_JOB_FAILURE, **self._job_msg_attrs(job)},
                    {"error_msg": err},
                )

            case Exited(job, error_message=None):
                logger.debug(f"Job {job.name} exited successfully")
                self._dump_event(
                    {_JOB_MSG_TYPE: _FM_JOB_SUCCESS, **self._job_msg_attrs(job)}
                )
            case Exited(job, exit_code, error_message=err):
                logger.error(
                    _JOB_EXIT_FAILED_STRING.format(
                        job_name=job.name(), exit_code=exit_code, error_message=err
                    )
                )
                self._dump_event(
                    {_JOB_MSG_TYPE: _FM_JOB_FAILURE, **self._job_msg_attrs(job)},
                    {"exit_code": msg.exit_code, "error_msg": msg.error_message},
                )

            case Running(job, error_message=err):
                logger.debug(f"{job.name} job is running")
                self._dump_event(
                    {_JOB_MSG_TYPE: _FM_JOB_RUNNING, **self._job_msg_attrs(job)},
                    {
                        "max_memory_usage": msg.max_memory_usage,
                        "current_memory_usage": msg.current_memory_usage,
                    },
                )

    def _finished_handler(self, msg):
        self._event_queue.put(self._sentinel)
        with self._timestamp_lock:
            self._timeout_timestamp = datetime.datetime.now() + datetime.timedelta(
                seconds=self._reporter_timeout
            )
        if self._event_publisher_thread.is_alive():
            self._event_publisher_thread.join()
