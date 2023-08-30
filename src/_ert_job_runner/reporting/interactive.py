from typing import Optional

from _ert_job_runner.reporting.base import Reporter
from _ert_job_runner.reporting.message import (
    _JOB_EXIT_FAILED_STRING,
    Finish,
    Message,
    Start,
)


class Interactive(Reporter):
    def _report(self, msg: Message) -> Optional[str]:
        job = msg.job
        name = job.name() if job is not None else "unknown"
        if not isinstance(msg, (Start, Finish)):
            return None
        if isinstance(msg, Finish):
            return (
                "OK"
                if msg.success()
                else _JOB_EXIT_FAILED_STRING.format(
                    job_name=name,
                    exit_code="No Code",
                    error_message=msg.error_message,
                )
            )
        return f"Running job: {name} ... "

    def report(self, msg: Message) -> None:
        _msg = self._report(msg)
        if _msg is not None:
            print(_msg)
