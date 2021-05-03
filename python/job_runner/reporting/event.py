from cloudevents.http import CloudEvent, to_json
from job_runner.reporting.message import (
    Exited,
    Finish,
    Init,
    Running,
    Start,
)
import queue
import threading
from pathlib import Path
from job_runner.util.client import Client

_FM_JOB_START = "com.equinor.ert.forward_model_job.start"
_FM_JOB_RUNNING = "com.equinor.ert.forward_model_job.running"
_FM_JOB_SUCCESS = "com.equinor.ert.forward_model_job.success"
_FM_JOB_FAILURE = "com.equinor.ert.forward_model_job.failure"


class TransitionError(ValueError):
    pass


class Event:
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
        with Client(self._evaluator_url, self._token, self._cert) as client:
            while True:
                event = self._event_queue.get()
                if event is None:
                    return
                client.send(to_json(event).decode())

    def _initialize_state_machine(self):
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
            raise TransitionError(
                f"Illegal transition {self._state} -> {new_state} for {msg}, expected to transition into {self._transitions[self._state]}"
            )

        self._states[new_state](msg)
        self._state = new_state

    def _dump_event(self, event):
        self._event_queue.put(event)

    def _step_path(self):
        return f"/ert/ee/{self._ee_id}/real/{self._real_id}/step/{self._step_id}"

    def _init_handler(self, msg):
        self._ee_id = msg.ee_id
        self._real_id = msg.real_id
        self._step_id = msg.step_id
        self._event_publisher_thread.start()

    def _job_handler(self, msg):
        job_path = f"{self._step_path()}/job/{msg.job.index}"

        if isinstance(msg, Start):
            self._dump_event(
                CloudEvent(
                    {
                        "type": _FM_JOB_START,
                        "source": job_path,
                        "datacontenttype": "application/json",
                    },
                    {
                        "stdout": str(Path(msg.job.std_out).resolve()),
                        "stderr": str(Path(msg.job.std_err).resolve()),
                    },
                )
            )
            if not msg.success():
                self._dump_event(
                    CloudEvent(
                        {
                            "type": _FM_JOB_FAILURE,
                            "source": job_path,
                            "datacontenttype": "application/json",
                        },
                        {
                            "error_msg": msg.error_message,
                        },
                    )
                )

        elif isinstance(msg, Exited):
            if msg.success():
                self._dump_event(
                    CloudEvent(
                        {
                            "type": _FM_JOB_SUCCESS,
                            "source": job_path,
                        },
                        None,
                    )
                )
            else:
                self._dump_event(
                    CloudEvent(
                        {
                            "type": _FM_JOB_FAILURE,
                            "source": job_path,
                            "datacontenttype": "application/json",
                        },
                        {
                            "exit_code": msg.exit_code,
                            "error_msg": msg.error_message,
                        },
                    )
                )

        elif isinstance(msg, Running):
            self._dump_event(
                CloudEvent(
                    {
                        "type": _FM_JOB_RUNNING,
                        "source": job_path,
                        "datacontenttype": "application/json",
                    },
                    {
                        "max_memory_usage": msg.max_memory_usage,
                        "current_memory_usage": msg.current_memory_usage,
                    },
                )
            )

    def _finished_handler(self, msg):
        self._event_queue.put(None)
        self._event_publisher_thread.join()
