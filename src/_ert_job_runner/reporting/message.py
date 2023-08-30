from __future__ import annotations

from datetime import datetime as dt
from typing import List, Optional, Union

from typing_extensions import Self

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


class _MetaMessage(type):
    def __repr__(cls) -> str:
        return f"MessageType<{cls.__name__}>"


class Message(metaclass=_MetaMessage):
    def __init__(self, job: Optional[Job] = None) -> None:
        self.timestamp = dt.now()
        self.job = job
        self.error_message: Optional[str] = None

    def __repr__(self) -> str:
        return type(self).__name__

    def with_error(self, message: str) -> Self:
        self.error_message = message
        return self

    def success(self) -> bool:
        return self.error_message is None


# manager level messages


class Init(Message):
    def __init__(
        self,
        jobs: List[Job],
        run_id: int,
        ert_pid: int,
        ens_id: Optional[str] = None,
        real_id: Optional[int] = None,
        step_id: Optional[int] = None,
        experiment_id: Optional[str] = None,
    ) -> None:
        super().__init__()
        self.jobs = jobs
        self.run_id = run_id
        self.ert_pid = ert_pid
        self.experiment_id = experiment_id
        self.ens_id = ens_id
        self.real_id = real_id
        self.step_id = step_id


class Finish(Message):
    def __init__(self) -> None:
        super().__init__()


# job level messages


class Start(Message):
    def __init__(self, job: Job) -> None:
        super().__init__(job)


class Running(Message):
    def __init__(
        self, job: Job, max_memory_usage: int, current_memory_usage: int
    ) -> None:
        super().__init__(job)
        self.max_memory_usage = max_memory_usage
        self.current_memory_usage = current_memory_usage


class Exited(Message):
    def __init__(self, job: Job, exit_code: Optional[int]) -> None:
        super().__init__(job)
        self.exit_code = exit_code
