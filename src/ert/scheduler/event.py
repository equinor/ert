from __future__ import annotations

from dataclasses import dataclass


@dataclass
class StartedEvent:
    iens: int
    exec_hosts: str = "-"


@dataclass
class FinishedEvent:
    iens: int
    returncode: int
    exec_hosts: str = "-"


@dataclass
class SchedulerWarningEvent:
    """This event is to indicate that something unexpected happened while
    running the ensemble, and that it might be stuck in an unresponsive state.
    """

    warning_message: str


DriverEvent = StartedEvent | FinishedEvent | SchedulerWarningEvent
