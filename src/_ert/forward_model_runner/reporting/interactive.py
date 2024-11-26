from typing import Optional

from _ert.forward_model_runner.reporting.base import Reporter
from _ert.forward_model_runner.reporting.message import (
    _JOB_EXIT_FAILED_STRING,
    Finish,
    Message,
    Start,
)


class Interactive(Reporter):
    @staticmethod
    def _report(msg: Message) -> Optional[str]:
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

    async def report(self, msg: Message):
        _msg = self._report(msg)
        if _msg is not None:
            print(_msg)

    def cancel(self):
        pass
