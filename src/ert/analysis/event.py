from dataclasses import dataclass


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
