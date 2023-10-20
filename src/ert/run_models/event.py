from dataclasses import dataclass


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
    pass
