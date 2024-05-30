from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Dict, List, Union


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
class RunModelCSVEvent(RunModelEvent):
    name: str
    header: List[str]
    data: Sequence[Sequence[Union[str, float]]]


@dataclass
class RunModelUpdateEndEvent(RunModelCSVEvent):
    extra: Dict[str, str]


@dataclass
class RunModelErrorEvent(RunModelUpdateEndEvent):
    error_msg: str
