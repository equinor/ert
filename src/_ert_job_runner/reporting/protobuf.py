import logging
import queue
import threading
from pathlib import Path
from typing import Union

from _ert_com_protocol import (
    JOB_FAILURE,
    JOB_RUNNING,
    JOB_START,
    JOB_SUCCESS,
    DispatcherMessage,
    EnsembleId,
    ExperimentId,
    Job,
    JobId,
    RealizationId,
    StepId,
)
from _ert_job_runner.client import Client
from _ert_job_runner.reporting.base import Reporter
from _ert_job_runner.reporting.message import (
    _JOB_EXIT_FAILED_STRING,
    Exited,
    Finish,
    Init,
    Running,
    Start,
)
from _ert_job_runner.reporting.statemachine import StateMachine

logger = logging.getLogger(__name__)


class Protobuf(Reporter):
    def __init__(self, experiment_url, token=None, cert_path=None):
        self._url = experiment_url
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

        self._experiment_id = None
        self._ens_id = None
        self._real_id = None
        self._step_id = None
        self._event_queue = queue.Queue()
        self._event_publisher_thread = threading.Thread(target=self._publish_event)

    def _dump_event(self, msg):
        if msg is None:
            self._event_queue.put(None)
        else:
            message = DispatcherMessage(job=msg)
            self._event_queue.put(message.SerializeToString())

    def _publish_event(self):
        logger.debug("Publishing event.")
        with Client(self._url, self._token, self._cert) as client:
            while True:
                event = self._event_queue.get()
                if event is None:
                    return
                client.send(event)

    def report(self, msg):
        self._statemachine.transition(msg)

    def _init_handler(self, msg: Init):
        self._experiment_id = msg.experiment_id
        self._ens_id = msg.ens_id
        self._real_id = int(msg.real_id)
        self._step_id = int(msg.step_id)
        self._event_publisher_thread.start()

    def _job_handler(self, msg: Union[Start, Running, Exited]):
        job_name = msg.job.name()
        pjob = Job(
            id=JobId(
                index=msg.job.index,
                step=StepId(
                    step=self._step_id,
                    realization=RealizationId(
                        realization=self._real_id,
                        ensemble=EnsembleId(
                            id=self._ens_id,
                            experiment=ExperimentId(id=self._experiment_id),
                        ),
                    ),
                ),
            )
        )
        if isinstance(msg, Start):
            logger.debug(f"Job {job_name} was successfully started")
            pjob.status = JOB_START
            pjob.stdout = str(Path(msg.job.std_out).resolve())
            pjob.stderr = str(Path(msg.job.std_err).resolve())
            self._dump_event(pjob)
            if not msg.success():
                logger.error(f"Job {job_name} FAILED to start")
                pjob.status = JOB_FAILURE
                pjob.error = msg.error_message
                self._dump_event(pjob)

        elif isinstance(msg, Exited):
            if msg.success():
                logger.debug(f"Job {job_name} exited successfully")
                pjob.status = JOB_SUCCESS
            else:
                logger.error(
                    _JOB_EXIT_FAILED_STRING.format(
                        job_name=msg.job.name(),
                        exit_code=msg.exit_code,
                        error_message=msg.error_message,
                    )
                )
                pjob.status = JOB_FAILURE
                pjob.exit_code = msg.exit_code
                pjob.error = msg.error_message
            self._dump_event(pjob)

        elif isinstance(msg, Running):
            logger.debug(f"{job_name} job is running")
            pjob.status = JOB_RUNNING
            pjob.current_memory = msg.current_memory_usage
            pjob.max_memory = msg.max_memory_usage
            self._dump_event(pjob)

    def _finished_handler(self, msg):
        self._dump_event(None)
        self._event_publisher_thread.join()
