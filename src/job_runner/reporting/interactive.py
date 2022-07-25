from typing import Optional

from job_runner.reporting.message import Message, Finish, Start, _JOB_EXIT_FAILED_STRING
from job_runner.reporting.base import Reporter


class Interactive(Reporter):
    def _report(self, msg: Message) -> Optional[str]:
        if not isinstance(msg, (Start, Finish)):
            return None
        if isinstance(msg, Finish):
            return (
                "OK"
                if msg.success()
                else _JOB_EXIT_FAILED_STRING.format(
                    job_name=msg.job.name(),
                    exit_code="No Code",
                    error_message=msg.error_message,
                )
            )
        return f"Running job: {msg.job.name()} ... "

    def report(self, msg: Message):
        _msg = self._report(msg)
        if _msg is not None:
            print(_msg)
