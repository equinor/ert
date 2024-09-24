from __future__ import annotations

from dataclasses import dataclass
from typing import Union


@dataclass
class StartedEvent:
    iens: int
    exec_hosts: str = "-"


@dataclass
class FinishedEvent:
    iens: int
    returncode: int
    exec_hosts: str = "-"


Event = Union[StartedEvent, FinishedEvent]
