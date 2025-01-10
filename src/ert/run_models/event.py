from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path
from uuid import UUID

from ert.analysis import (
    AnalysisEvent,
    AnalysisStatusEvent,
    AnalysisTimeEvent,
)
from ert.analysis.event import DataSection
from ert.ensemble_evaluator.event import (
    EndEvent,
    FullSnapshotEvent,
    SnapshotUpdateEvent,
)
from ert.ensemble_evaluator.snapshot import EnsembleSnapshot


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

    def write_as_csv(self, output_path: Path | None) -> None:
        if output_path and self.data:
            self.data.to_csv(self.name, output_path / str(self.run_id))


@dataclass
class RunModelUpdateEndEvent(RunModelEvent):
    data: DataSection

    def write_as_csv(self, output_path: Path | None) -> None:
        if output_path and self.data:
            self.data.to_csv("Report", output_path / str(self.run_id))


@dataclass
class RunModelErrorEvent(RunModelEvent):
    error_msg: str
    data: DataSection | None = None

    def write_as_csv(self, output_path: Path | None) -> None:
        if output_path and self.data:
            self.data.to_csv("Report", output_path / str(self.run_id))


StatusEvents = (
    AnalysisEvent
    | AnalysisStatusEvent
    | AnalysisTimeEvent
    | EndEvent
    | FullSnapshotEvent
    | SnapshotUpdateEvent
    | RunModelErrorEvent
    | RunModelStatusEvent
    | RunModelTimeEvent
    | RunModelUpdateBeginEvent
    | RunModelDataEvent
    | RunModelUpdateEndEvent
)


EVENT_MAPPING = {
    "AnalysisEvent": AnalysisEvent,
    "AnalysisStatusEvent": AnalysisStatusEvent,
    "AnalysisTimeEvent": AnalysisTimeEvent,
    "EndEvent": EndEvent,
    "FullSnapshotEvent": FullSnapshotEvent,
    "SnapshotUpdateEvent": SnapshotUpdateEvent,
    "RunModelErrorEvent": RunModelErrorEvent,
    "RunModelStatusEvent": RunModelStatusEvent,
    "RunModelTimeEvent": RunModelTimeEvent,
    "RunModelUpdateBeginEvent": RunModelUpdateBeginEvent,
    "RunModelDataEvent": RunModelDataEvent,
    "RunModelUpdateEndEvent": RunModelUpdateEndEvent,
}


def status_event_from_json(json_str: str) -> StatusEvents:
    json_dict = json.loads(json_str)
    event_type = json_dict.pop("event_type", None)

    match event_type:
        case FullSnapshotEvent.__name__:
            snapshot = EnsembleSnapshot.from_nested_dict(json_dict["snapshot"])
            json_dict["snapshot"] = snapshot
            return FullSnapshotEvent(**json_dict)
        case SnapshotUpdateEvent.__name__:
            snapshot = EnsembleSnapshot.from_nested_dict(json_dict["snapshot"])
            json_dict["snapshot"] = snapshot
            return SnapshotUpdateEvent(**json_dict)
        case RunModelDataEvent.__name__ | RunModelUpdateEndEvent.__name__:
            if "run_id" in json_dict and isinstance(json_dict["run_id"], str):
                json_dict["run_id"] = UUID(json_dict["run_id"])
            if json_dict.get("data"):
                json_dict["data"] = DataSection(**json_dict["data"])
            return EVENT_MAPPING[event_type](**json_dict)
        case _:
            if event_type in EVENT_MAPPING:
                if "run_id" in json_dict and isinstance(json_dict["run_id"], str):
                    json_dict["run_id"] = UUID(json_dict["run_id"])
                return EVENT_MAPPING[event_type](**json_dict)
            else:
                raise TypeError(f"Unknown status event type {event_type}")


def status_event_to_json(event: StatusEvents) -> str:
    match event:
        case FullSnapshotEvent() | SnapshotUpdateEvent():
            assert event.snapshot is not None
            event_dict = asdict(event)
            event_dict.update(
                {
                    "snapshot": event.snapshot.to_dict(),
                    "event_type": event.__class__.__name__,
                }
            )
            return json.dumps(
                event_dict,
                default=lambda o: o.strftime("%Y-%m-%dT%H:%M:%S")
                if isinstance(o, datetime)
                else None,
            )
        case StatusEvents:
            event_dict = asdict(event)
            event_dict["event_type"] = StatusEvents.__class__.__name__
            return json.dumps(
                event_dict, default=lambda o: str(o) if isinstance(o, UUID) else None
            )
