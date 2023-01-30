from dataclasses import dataclass
from enum import Enum


class StatusEnum(Enum):
    RUNNING = "RUNNING"
    FINISHED = "FINISHED"
    CANCELED = "CANCELED"
    FAILED = "FAILED"
    NOT_STARTED = "NOT_STARTED"


@dataclass
class RunStatus:
    status: StatusEnum = StatusEnum.NOT_STARTED
    stdoutdata: str = ""
    stderrdata: str = ""

    def start(self):
        self.status = StatusEnum.RUNNING

    def cancel(self):
        self.status = StatusEnum.CANCELED

    def finish(self, msg=""):
        if isinstance(msg, bytes):
            msg = msg.decode()
        self.stdoutdata = msg
        if self.status == StatusEnum.RUNNING:
            self.status = StatusEnum.FINISHED

    def error(self, msg=""):
        if isinstance(msg, bytes):
            msg = msg.decode()
        self.status = StatusEnum.FAILED
        self.stderrdata = msg

    def has_finished(self):
        return self.status == StatusEnum.FINISHED

    def was_canceled(self):
        return self.status == StatusEnum.CANCELED

    def has_failed(self):
        return self.status == StatusEnum.FAILED

    def is_running(self):
        return self.status == StatusEnum.RUNNING

    def not_started(self):
        return self.status == StatusEnum.NOT_STARTED
