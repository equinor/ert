from __future__ import annotations

from dataclasses import dataclass
from typing import Union


@dataclass
class StartedEvent:
    iens: int


@dataclass
class FinishedEvent:
    iens: int
    returncode: int


Event = Union[StartedEvent, FinishedEvent]
