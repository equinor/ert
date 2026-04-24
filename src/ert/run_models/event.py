from __future__ import annotations

from pathlib import Path
from typing import Annotated, Any, Literal
from uuid import UUID

from pydantic import BaseModel, ConfigDict, Field, TypeAdapter

from _ert.events import EnsembleEvaluationWarning
from ert.analysis import (
    AnalysisStatusEvent,
    AnalysisTimeEvent,
)
from ert.analysis.event import DataSection
from ert.ensemble_evaluator.event import (
    EndEvent,
    FullSnapshotEvent,
    SnapshotUpdateEvent,
    StartEvent,
    WarningEvent,
)


class RunModelEvent(BaseModel):
    model_config = ConfigDict(arbitrary_types_allowed=True, extra="forbid")
    iteration: int
    run_id: UUID


class RunModelStatusEvent(RunModelEvent):
    event_type: Literal["RunModelStatusEvent"] = "RunModelStatusEvent"
    msg: str
    detail: bool = False


class EverestStatusEvent(BaseModel):
    batch: int | None
    event_type: Literal["EverestStatusEvent"] = "EverestStatusEvent"
    everest_event: Literal[
        "START_OPTIMIZER_EVALUATION",
        "FINISHED_OPTIMIZER_EVALUATION",
    ]


class EverestBatchResultEvent(BaseModel):
    batch: int
    event_type: Literal["EverestBatchResultEvent"] = "EverestBatchResultEvent"
    everest_event: Literal["OPTIMIZATION_RESULT",]
    result_type: Literal["FunctionResult", "GradientResult"]
    results: dict[str, Any] | None = None


class RunModelTimeEvent(RunModelEvent):
    event_type: Literal["RunModelTimeEvent"] = "RunModelTimeEvent"
    remaining_time: float
    elapsed_time: float


class RunModelUpdateBeginEvent(RunModelEvent):
    event_type: Literal["RunModelUpdateBeginEvent"] = "RunModelUpdateBeginEvent"


class RunModelDataEvent(RunModelEvent):
    event_type: Literal["RunModelDataEvent"] = "RunModelDataEvent"
    name: str
    data: DataSection

    def write_as_csv(self, output_path: Path | None) -> None:
        if output_path and self.data:
            self.data.to_csv(self.name, output_path / str(self.run_id))


class RunModelMatrixEvent(RunModelEvent):
    event_type: Literal["RunModelMatrixEvent"] = "RunModelMatrixEvent"
    name: str


class RunModelUpdateEndEvent(RunModelEvent):
    event_type: Literal["RunModelUpdateEndEvent"] = "RunModelUpdateEndEvent"
    data: DataSection

    def write_as_csv(self, output_path: Path | None) -> None:
        if output_path and self.data:
            self.data.to_csv("Report", output_path / str(self.run_id))


class RunModelErrorEvent(RunModelEvent):
    event_type: Literal["RunModelErrorEvent"] = "RunModelErrorEvent"
    error_msg: str
    data: DataSection | None

    def write_as_csv(self, output_path: Path | None) -> None:
        if output_path and self.data:
            self.data.to_csv("Report", output_path / str(self.run_id))


class RunPathCreationEvent(BaseModel):
    pass


class StartingTotalRunPathCreationEvent(RunPathCreationEvent):
    event_type: Literal["StartingTotalRunPathCreation"] = "StartingTotalRunPathCreation"
    total_runpaths_to_create: int


class FinishedTotalRunPathCreationEvent(RunPathCreationEvent):
    event_type: Literal["FinishedTotalRunPathCreationEvent"] = (
        "FinishedTotalRunPathCreationEvent"
    )


class RunPathCreatedEvent(RunPathCreationEvent):
    event_type: Literal["RunPathCreatedEvent"] = "RunPathCreatedEvent"
    iens: int


StatusEvents = (
    AnalysisStatusEvent
    | AnalysisTimeEvent
    | EndEvent
    | EverestBatchResultEvent
    | EverestStatusEvent
    | FullSnapshotEvent
    | RunModelDataEvent
    | RunModelErrorEvent
    | RunModelMatrixEvent
    | RunModelStatusEvent
    | RunModelTimeEvent
    | RunModelUpdateBeginEvent
    | RunModelUpdateEndEvent
    | SnapshotUpdateEvent
    | StartEvent
    | WarningEvent
    | EnsembleEvaluationWarning
    | StartingTotalRunPathCreationEvent
    | FinishedTotalRunPathCreationEvent
    | RunPathCreatedEvent
)


STATUS_EVENTS_ANNOTATION = Annotated[StatusEvents, Field(discriminator="event_type")]

StatusEventAdapter: TypeAdapter[StatusEvents] = TypeAdapter(STATUS_EVENTS_ANNOTATION)


def status_event_from_json(raw_msg: str | bytes) -> StatusEvents:
    return StatusEventAdapter.validate_json(raw_msg)


def status_event_to_json(event: StatusEvents) -> str:
    return event.model_dump_json()
