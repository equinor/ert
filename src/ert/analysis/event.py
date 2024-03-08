from dataclasses import dataclass
from typing import Optional

from .snapshots import SmootherSnapshot


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
class AnalysisErrorEvent(AnalysisEvent):
    smoother_snapshot: Optional[SmootherSnapshot] = None
    error_msg: Optional[str] = None
