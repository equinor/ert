from __future__ import annotations

from pathlib import Path
from typing import Annotated, Literal
from uuid import UUID

from pydantic import BaseModel, ConfigDict, Field, TypeAdapter

from ert.analysis import (
    AnalysisStatusEvent,
    AnalysisTimeEvent,
)
from ert.analysis.event import DataSection
from ert.ensemble_evaluator.event import (
    EndEvent,
    FullSnapshotEvent,
    SnapshotUpdateEvent,
)


class RunModelEvent(BaseModel):
    model_config = ConfigDict(arbitrary_types_allowed=True, extra="forbid")
    iteration: int
    run_id: UUID


class RunModelStatusEvent(RunModelEvent):
    event_type: Literal["RunModelStatusEvent"] = "RunModelStatusEvent"
    msg: str


class RunModelTimeEvent(RunModelEvent):
    event_type: Literal["RunModelTimeEvent"] = "RunModelTimeEvent"
    remaining_time: float
    elapsed_time: float


class RunModelUpdateBeginEvent(RunModelEvent):
    event_type: Literal["RunModelUpdateBeginEvent"] = "RunModelUpdateBeginEvent"
    pass


class RunModelDataEvent(RunModelEvent):
    event_type: Literal["RunModelDataEvent"] = "RunModelDataEvent"
    name: str
    data: DataSection

    def write_as_csv(self, output_path: Path | None) -> None:
        if output_path and self.data:
            self.data.to_csv(self.name, output_path / str(self.run_id))


class RunModelUpdateEndEvent(RunModelEvent):
    event_type: Literal["RunModelUpdateEndEvent"] = "RunModelUpdateEndEvent"
    data: DataSection

    def write_as_csv(self, output_path: Path | None) -> None:
        if output_path and self.data:
            self.data.to_csv("Report", output_path / str(self.run_id))


class RunModelErrorEvent(RunModelEvent):
    event_type: Literal["RunModelErrorEvent"] = "RunModelErrorEvent"
    error_msg: str
    data: DataSection

    def write_as_csv(self, output_path: Path | None) -> None:
        if output_path and self.data:
            self.data.to_csv("Report", output_path / str(self.run_id))


StatusEvents = (
    AnalysisStatusEvent
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


STATUS_EVENTS_ANNOTATION = Annotated[StatusEvents, Field(discriminator="event_type")]

StatusEventAdapter: TypeAdapter[StatusEvents] = TypeAdapter(STATUS_EVENTS_ANNOTATION)


def status_event_from_json(raw_msg: str | bytes) -> StatusEvents:
    return StatusEventAdapter.validate_json(raw_msg)


def status_event_to_json(event: StatusEvents) -> str:
    return event.model_dump_json()
