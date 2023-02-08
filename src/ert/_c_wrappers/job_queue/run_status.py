from dataclasses import dataclass
from enum import Enum, auto


class RunStatus(Enum):
    RUNNING = auto()
    FINISHED = auto()
    CANCELED = auto()
    FAILED = auto()
    NOT_STARTED = auto()


@dataclass
class RunTracker:
    status: RunStatus = RunStatus.NOT_STARTED
    stdoutdata: str = ""
    stderrdata: str = ""

    def start(self):
        self.status = RunStatus.RUNNING

    def cancel(self):
        self.status = RunStatus.CANCELED

    def finish(self, msg=""):
        if isinstance(msg, bytes):
            msg = msg.decode()
        self.stdoutdata = msg
        if self.status == RunStatus.RUNNING:
            self.status = RunStatus.FINISHED

    def error(self, msg=""):
        if isinstance(msg, bytes):
            msg = msg.decode()
        self.status = RunStatus.FAILED
        self.stderrdata = msg

    def has_finished(self):
        return self.status == RunStatus.FINISHED

    def was_canceled(self):
        return self.status == RunStatus.CANCELED

    def has_failed(self):
        return self.status == RunStatus.FAILED

    def is_running(self):
        return self.status == RunStatus.RUNNING

    def not_started(self):
        return self.status == RunStatus.NOT_STARTED
