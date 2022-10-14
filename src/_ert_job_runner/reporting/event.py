import logging
import queue
import threading
from pathlib import Path
from typing import Any, Dict

from cloudevents.conversion import to_json
from cloudevents.http import CloudEvent

from _ert_job_runner.client import Client
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
    # pylint: disable=too-many-instance-attributes
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
        self._step_id = None
        self._event_queue = queue.Queue()
        self._event_publisher_thread = threading.Thread(target=self._publish_event)

    def _publish_event(self):
        logger.debug("Publishing event.")
        with Client(self._evaluator_url, self._token, self._cert) as client:
            while True:
                event = self._event_queue.get()
                if event is None:
                    return
                client.send(to_json(event).decode())

    def report(self, msg):
        self._statemachine.transition(msg)

    def _dump_event(self, attributes: Dict[str, str], data: Any = None):
        if data is None and _CONTENT_TYPE in attributes:
            attributes.pop(_CONTENT_TYPE)

        event = CloudEvent(attributes=attributes, data=data)
        logger.debug(f'Schedule {type(event)} "{event["type"]}" for delivery')
        self._event_queue.put(event)

    def _step_path(self):
        return f"/ert/ensemble/{self._ens_id}/real/{self._real_id}/step/{self._step_id}"

    def _init_handler(self, msg):
        self._ens_id = msg.ens_id
        self._real_id = msg.real_id
        self._step_id = msg.step_id
        self._event_publisher_thread.start()

    def _job_handler(self, msg: Message):
        job_name = msg.job.name()
        job_msg_attrs = {
            _JOB_SOURCE: (
                f"{self._step_path()}/job/{msg.job.index}" f"/index/{msg.job.index}"
            ),
            _CONTENT_TYPE: "application/json",
        }
        if isinstance(msg, Start):
            logger.debug(f"Job {job_name} was successfully started")
            self._dump_event(
                attributes={_JOB_MSG_TYPE: _FM_JOB_START, **job_msg_attrs},
                data={
                    "stdout": str(Path(msg.job.std_out).resolve()),
                    "stderr": str(Path(msg.job.std_err).resolve()),
                },
            )
            if not msg.success():
                logger.error(f"Job {job_name} FAILED to start")
                self._dump_event(
                    attributes={_JOB_MSG_TYPE: _FM_JOB_FAILURE, **job_msg_attrs},
                    data={
                        "error_msg": msg.error_message,
                    },
                )

        elif isinstance(msg, Exited):
            data = None
            if msg.success():
                logger.debug(f"Job {job_name} exited successfully")
                attributes = {_JOB_MSG_TYPE: _FM_JOB_SUCCESS, **job_msg_attrs}
            else:
                logger.error(
                    _JOB_EXIT_FAILED_STRING.format(
                        job_name=msg.job.name(),
                        exit_code=msg.exit_code,
                        error_message=msg.error_message,
                    )
                )
                attributes = {_JOB_MSG_TYPE: _FM_JOB_FAILURE, **job_msg_attrs}
                data = {
                    "exit_code": msg.exit_code,
                    "error_msg": msg.error_message,
                }
            self._dump_event(attributes=attributes, data=data)

        elif isinstance(msg, Running):
            logger.debug(f"{job_name} job is running")
            self._dump_event(
                attributes={_JOB_MSG_TYPE: _FM_JOB_RUNNING, **job_msg_attrs},
                data={
                    "max_memory_usage": msg.max_memory_usage,
                    "current_memory_usage": msg.current_memory_usage,
                },
            )

    def _finished_handler(self, msg):
        self._event_queue.put(None)
        self._event_publisher_thread.join()
