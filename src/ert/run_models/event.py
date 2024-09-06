from __future__ import annotations

from dataclasses import dataclass
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


@dataclass
class RunModelTimeEvent(RunModelEvent):
    remaining_time: float
    elapsed_time: float
    event_type: Literal['RunModelTimeEvent'] = 'RunModelTimeEvent'


@dataclass
class RunModelUpdateBeginEvent(RunModelEvent):
    event_type: Literal['RunModelUpdateBeginEvent'] = 'RunModelUpdateBeginEvent'


@dataclass
class RunModelDataEvent(RunModelEvent):
    name: str
    data: DataSection
    event_type: Literal['RunModelDataEvent'] = 'RunModelDataEvent'


@dataclass
class RunModelUpdateEndEvent(RunModelEvent):
    data: DataSection
    event_type: Literal['RunModelUpdateEndEvent'] = 'RunModelUpdateEndEvent'


@dataclass
class RunModelErrorEvent(RunModelEvent):
    error_msg: str
    data: Optional[DataSection] = None
    event_type: Literal['RunModelErrorEvent'] = 'RunModelErrorEvent'
