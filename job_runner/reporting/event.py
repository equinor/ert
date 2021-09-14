import queue
import threading
import logging
from pathlib import Path
from typing import Dict, Any

from cloudevents.http import CloudEvent, to_json

from job_runner.reporting.message import (
    Exited,
    Finish,
    Init,
    Running,
    Start,
    Message,
    _JOB_EXIT_FAILED_STRING,
)
from job_runner.reporting.base import Reporter
from job_runner.util.client import Client

_FM_JOB_START = "com.equinor.ert.forward_model_job.start"
_FM_JOB_RUNNING = "com.equinor.ert.forward_model_job.running"
_FM_JOB_SUCCESS = "com.equinor.ert.forward_model_job.success"
_FM_JOB_FAILURE = "com.equinor.ert.forward_model_job.failure"

_CONTENT_TYPE = "datacontenttype"
_JOB_MSG_TYPE = "type"
_JOB_SOURCE = "source"

logger = logging.getLogger(__name__)


class TransitionError(ValueError):
    pass


class Event(Reporter):
    def __init__(self, evaluator_url, token=None, cert_path=None):
        self._evaluator_url = evaluator_url
        self._token = token
        if cert_path is not None:
            with open(cert_path) as f:
                self._cert = f.read()
        else:
            self._cert = None

        self._ee_id = None
        self._real_id = None
        self._step_id = None
        self._event_queue = queue.Queue()
        self._event_publisher_thread = threading.Thread(target=self._publish_event)
        self._initialize_state_machine()

    def _publish_event(self):
        logger.debug("Publishing event.")
        with Client(self._evaluator_url, self._token, self._cert) as client:
            while True:
                event = self._event_queue.get()
                if event is None:
                    return
                client.send(to_json(event).decode())

    def _initialize_state_machine(self):
        logger.debug("Initializing state machines")
        initialized = (Init,)
        jobs = (Start, Running, Exited)
        finished = (Finish,)
        self._states = {
            initialized: self._init_handler,
            jobs: self._job_handler,
            finished: self._finished_handler,
        }
        self._transitions = {
            None: initialized,
            initialized: jobs + finished,
            jobs: jobs + finished,
        }
        self._state = None

    def report(self, msg):
        new_state = None
        for state in self._states.keys():
            if isinstance(msg, state):
                new_state = state

        if self._state not in self._transitions or not isinstance(
            msg, self._transitions[self._state]
        ):
            logger.error(
                f"{msg} illegal state transition: {self._state} -> {new_state}"
            )
            raise TransitionError(
                f"Illegal transition {self._state} -> {new_state} for {msg}, expected to transition into {self._transitions[self._state]}"
            )

        self._states[new_state](msg)
        self._state = new_state

    def _dump_event(self, attributes: Dict[str, str], data: Any = None):
        if data is None and _CONTENT_TYPE in attributes:
            attributes.pop(_CONTENT_TYPE)

        event = CloudEvent(attributes=attributes, data=data)
        logger.debug(f'Schedule {type(event)} "{event["type"]}" for delivery')
        self._event_queue.put(event)

    def _step_path(self):
        return f"/ert/ee/{self._ee_id}/real/{self._real_id}/step/{self._step_id}"

    def _init_handler(self, msg):
        self._ee_id = msg.ee_id
        self._real_id = msg.real_id
        self._step_id = msg.step_id
        self._event_publisher_thread.start()

    def _job_handler(self, msg: Message):
        job_name = msg.job.name()
        job_msg_attrs = {
            _JOB_SOURCE: f"{self._step_path()}/job/{msg.job.index}",
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
