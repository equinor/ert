from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime as dt
from typing import TYPE_CHECKING, List, Optional

if TYPE_CHECKING:
    from _ert_job_runner.job import Job

_JOB_STATUS_SUCCESS = "Success"
_JOB_STATUS_RUNNING = "Running"
_JOB_STATUS_FAILURE = "Failure"
_JOB_STATUS_WAITING = "Waiting"

_RUNNER_STATUS_INITIALIZED = "Initialized"
_RUNNER_STATUS_SUCCESS = "Success"
_RUNNER_STATUS_FAILURE = "Failure"


_JOB_EXIT_FAILED_STRING = """Job {job_name} FAILED with code {exit_code}
----------------------------------------------------------
Error message: {error_message}
----------------------------------------------------------
"""


@dataclass
class Message:
    timestamp: dt = field(default=dt.now(), kw_only=True)
    error_message: Optional[str] = field(default=None, kw_only=True)

    def __repr__(self):
        return type(self).__name__

    def with_error(self, message):
        self.error_message = message
        return self

    def success(self):
        return self.error_message is None


@dataclass
class Init(Message):
    jobs: List[Job]
    run_id: str
    ert_pid: Optional[int] = None
    ens_id: Optional[str] = None
    real_id: Optional[int] = None
    experiment_id: Optional[str] = None


@dataclass
class Finish(Message):
    pass


@dataclass
class Start(Message):
    job: Job


@dataclass
class Running(Message):
    job: Job
    max_memory_usage: int
    current_memory_usage: int


@dataclass
class Exited(Message):
    job: Job
    exit_code: Optional[int] = None
