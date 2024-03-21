from dataclasses import dataclass
from typing import Optional

from ert.analysis import SmootherSnapshot


@dataclass
class RunModelEvent:
    iteration: int


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
class RunModelUpdateEndEvent(RunModelEvent):
    smoother_snapshot: Optional[SmootherSnapshot] = None


@dataclass
class RunModelErrorEvent(RunModelEvent):
    smoother_snapshot: Optional[SmootherSnapshot] = None
    error_msg: Optional[str] = None
