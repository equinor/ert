from _ert.forward_model_runner.reporting.base import Reporter
from _ert.forward_model_runner.reporting.message import (
    STEP_EXIT_FAILED_STRING_TEMPLATE,
    Finish,
    Message,
    Start,
)


class Interactive(Reporter):
    @staticmethod
    def _report(msg: Message) -> str | None:
        if not isinstance(msg, Start | Finish):
            return None
        assert msg.step is not None
        if isinstance(msg, Finish):
            return (
                "OK"
                if msg.success()
                else STEP_EXIT_FAILED_STRING_TEMPLATE.format(
                    step_name=msg.step.name(),
                    exit_code="No Code",
                    error_message=msg.error_message,
                )
            )
        return f"Running step: {msg.step.name()} ... "

    def report(self, msg: Message) -> None:
        msg_ = self._report(msg)
        if msg_ is not None:
            print(msg_)
