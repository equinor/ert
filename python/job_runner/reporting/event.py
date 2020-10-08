from cloudevents.http import CloudEvent, to_json
from job_runner.reporting.message import (
    Exited,
    Finish,
    Init,
    Running,
    Start,
)

_FM_JOB_START = "com.equinor.ert.forward_model_job.start"
_FM_JOB_RUNNING = "com.equinor.ert.forward_model_job.running"
_FM_JOB_SUCCESS = "com.equinor.ert.forward_model_job.success"
_FM_JOB_FAILURE = "com.equinor.ert.forward_model_job.failure"

_FM_STEP_START = "com.equinor.ert.forward_model_step.start"
_FM_STEP_FAILURE = "com.equinor.ert.forward_model_step.failure"
_FM_STEP_SUCCESS = "com.equinor.ert.forward_model_step.success"


class TransitionError(ValueError):
    pass


class Event:
    def __init__(self, event_log="event_log"):
        self._event_log = event_log

        self._ee_id = None
        self._real_id = None
        self._stage_id = None

        self._initialize_state_machine()
        self._clear_log()

    def _initialize_state_machine(self):
        initialized = (Init,)
        jobs = (Start, Running, Exited)
        finished = (Finish,)
        self._states = {
            initialized: self._init_handler,
            jobs: self._job_handler,
            finished: self._end_handler,
        }
        self._transitions = {
            None: initialized,
            initialized: jobs + finished,
            jobs: jobs + finished,
        }
        self._state = None

    def _clear_log(self):
        with open(self._event_log, "w") as f:
            pass

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
        with open(self._event_log, "a") as el:
            el.write("{}\n".format(to_json(event).decode()))

    def _step_path(self):
        return f"/ert/ee/{self._ee_id}/real/{self._real_id}/stage/{self._stage_id}/step/{0}"

    def _init_handler(self, msg):
        self._ee_id = msg.ee_id
        self._real_id = msg.real_id
        self._stage_id = msg.stage_id
        self._dump_event(
            CloudEvent(
                {
                    "type": _FM_STEP_START,
                    "source": self._step_path(),
                    "datacontenttype": "application/json",
                },
                {
                    "jobs": [job.job_data for job in msg.jobs],
                },
            )
        )

    def _job_handler(self, msg):
        job_path = f"{self._step_path()}/job/{msg.job.index}"

        if isinstance(msg, Start):
            self._dump_event(
                CloudEvent(
                    {
                        "type": _FM_JOB_START,
                        "source": job_path,
                    },
                    None,
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

    def _end_handler(self, msg):
        step_path = self._step_path()
        if msg.success():
            self._dump_event(
                CloudEvent(
                    {
                        "type": _FM_STEP_SUCCESS,
                        "source": step_path,
                    }
                )
            )
        else:
            self._dump_event(
                CloudEvent(
                    {
                        "type": _FM_STEP_FAILURE,
                        "source": step_path,
                        "datacontenttype": "application/json",
                    },
                    {
                        "error_msg": msg.error_message,
                    },
                )
            )
