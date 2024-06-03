from __future__ import annotations

from dataclasses import dataclass
from typing import Optional
from uuid import UUID

from ert.analysis.event import DataSection


@dataclass
class RunModelEvent:
    iteration: int
    run_id: UUID


@dataclass
class RunModelStatusEvent(RunModelEvent):
    msg: str


@dataclass
class RunModelTimeEvent(RunModelEvent):
    remaining_time: float
    elapsed_time: float


@dataclass
class RunModelUpdateBeginEvent(RunModelEvent):
    pass


@dataclass
class RunModelDataEvent(RunModelEvent):
    name: str
    data: DataSection


@dataclass
class RunModelUpdateEndEvent(RunModelEvent):
    data: DataSection


@dataclass
class RunModelErrorEvent(RunModelEvent):
    error_msg: str
    data: Optional[DataSection] = None
