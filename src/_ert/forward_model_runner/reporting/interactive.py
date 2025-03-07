from _ert.forward_model_runner.reporting.base import Reporter
from _ert.forward_model_runner.reporting.message import (
    _STEP_EXIT_FAILED_STRING,
    Finish,
    Message,
    Start,
)


class Interactive(Reporter):
    @staticmethod
    def _report(msg: Message) -> str | None:
        if not isinstance(msg, Start | Finish):
            return None
        if isinstance(msg, Finish):
            return (
                "OK"
                if msg.success()
                else _STEP_EXIT_FAILED_STRING.format(
                    step_name=msg.step.name(),
                    exit_code="No Code",
                    error_message=msg.error_message,
                )
            )
        return f"Running step: {msg.step.name()} ... "

    def report(self, msg: Message):
        msg_ = self._report(msg)
        if msg_ is not None:
            print(msg_)
