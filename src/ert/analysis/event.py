from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Dict, List, Union


@dataclass
class AnalysisEvent:
    pass


@dataclass
class AnalysisStatusEvent(AnalysisEvent):
    msg: str


@dataclass
class AnalysisTimeEvent(AnalysisEvent):
    remaining_time: float
    elapsed_time: float


@dataclass
class AnalysisReportEvent(AnalysisEvent):
    report: str


@dataclass
class AnalysisCSVEvent(AnalysisEvent):
    name: str
    header: List[str]
    data: Sequence[Sequence[Union[str, float]]]

    def __post_init__(self) -> None:
        if len(self.header) != len(self.data[0]):
            raise ValueError(
                f"Header ({self.header}) must have same length as "
                f"number of columns ({len(self.data[0])})"
            )


@dataclass
class AnalysisErrorEvent(AnalysisCSVEvent):
    extra: Dict[str, str]
    error_msg: str
