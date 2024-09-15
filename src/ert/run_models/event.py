from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from typing import Optional, Literal
from uuid import UUID

from ert.analysis.event import DataSection


@dataclass
class RunModelEvent:
    iteration: int
    run_id: UUID


@dataclass
class RunModelStatusEvent(RunModelEvent):
    msg: str
    event_type: Literal['RunModelStatusEvent'] = 'RunModelStatusEvent'
    timestamp: datetime = field(default_factory=datetime.now)


@dataclass
class RunModelTimeEvent(RunModelEvent):
    remaining_time: float
    elapsed_time: float
    event_type: Literal['RunModelTimeEvent'] = 'RunModelTimeEvent'
    timestamp: datetime = field(default_factory=datetime.now)


@dataclass
class RunModelUpdateBeginEvent(RunModelEvent):
    event_type: Literal['RunModelUpdateBeginEvent'] = 'RunModelUpdateBeginEvent'
    timestamp: datetime = field(default_factory=datetime.now)


@dataclass
class RunModelDataEvent(RunModelEvent):
    name: str
    data: DataSection
    event_type: Literal['RunModelDataEvent'] = 'RunModelDataEvent'
    timestamp: datetime = field(default_factory=datetime.now)


@dataclass
class RunModelUpdateEndEvent(RunModelEvent):
    data: DataSection
    event_type: Literal['RunModelUpdateEndEvent'] = 'RunModelUpdateEndEvent'
    timestamp: datetime = field(default_factory=datetime.now)


@dataclass
class RunModelErrorEvent(RunModelEvent):
    error_msg: str
    data: Optional[DataSection] = None
    event_type: Literal['RunModelErrorEvent'] = 'RunModelErrorEvent'
    timestamp: datetime = field(default_factory=datetime.now)
