from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Dict, List, Optional, Union


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
class DataSection:
    header: List[str]
    data: Sequence[Sequence[Union[str, float]]]
    extra: Optional[Dict[str, str]] = None

    def __post_init__(self) -> None:
        if len(self.data) > 0 and len(self.header) != len(self.data[0]):
            raise ValueError(
                f"Header ({self.header}) must have same length as "
                f"number of columns ({len(self.data[0])})"
            )


@dataclass
class AnalysisDataEvent(AnalysisEvent):
    name: str
    data: DataSection


@dataclass
class AnalysisErrorEvent(AnalysisEvent):
    error_msg: str
    data: Optional[DataSection] = None


@dataclass
class AnalysisCompleteEvent(AnalysisEvent):
    data: DataSection
