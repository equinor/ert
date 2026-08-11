from __future__ import annotations

import logging
from datetime import datetime
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
from ert.workflow_runner import WorkflowJobStatus

logger = logging.getLogger(__name__)


class RunModelEvent(BaseModel):
    model_config = ConfigDict(arbitrary_types_allowed=True, extra="forbid")
    iteration: int
    run_id: UUID


class RunModelStatusEvent(RunModelEvent):
    event_type: Literal["RunModelStatusEvent"] = "RunModelStatusEvent"
    msg: str
    detail: bool = False


class EverestStatusEvent(BaseModel, extra="forbid"):
    batch: int | None
    event_type: Literal["EverestStatusEvent"] = "EverestStatusEvent"
    everest_event: Literal[
        "START_OPTIMIZER_EVALUATION",
        "FINISHED_OPTIMIZER_EVALUATION",
    ]


class EverestBatchResultEvent(BaseModel, extra="forbid"):
    batch: int
    event_type: Literal["EverestBatchResultEvent"] = "EverestBatchResultEvent"
    everest_event: Literal["OPTIMIZATION_RESULT",]
    result_type: Literal["FunctionResult", "GradientResult"]
    results: dict[str, Any] = {}
    failures: dict[int, list[int]] = {}


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


class WorkflowEvent(BaseModel, extra="forbid"):
    """The output of a single workflow job invocation."""

    event_type: Literal["WorkflowEvent"] = "WorkflowEvent"
    run_id: UUID
    hook: str
    workflow_name: str
    job_name: str
    job_index: int
    arguments: list[str]
    stdout: str
    stderr: str
    status: WorkflowJobStatus
    timestamp: datetime
    iteration: int | None = None


class RunPathCreationEvent(BaseModel, extra="forbid"):
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
    | RunModelStatusEvent
    | RunModelTimeEvent
    | RunModelUpdateBeginEvent
    | RunModelUpdateEndEvent
    | SnapshotUpdateEvent
    | StartEvent
    | WarningEvent
    | WorkflowEvent
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


class CorruptStatusSnapshotError(Exception):
    """Raised when a persisted status snapshot exists but cannot be loaded."""


def load_status_snapshot_event(path: Path) -> FullSnapshotEvent | None:
    if not path.is_file():
        return None
    try:
        event = status_event_from_json(path.read_bytes())
    except OSError as e:
        raise CorruptStatusSnapshotError(
            f"Could not read status snapshot from {path}"
        ) from e
    except ValueError as e:
        raise CorruptStatusSnapshotError(
            f"Could not parse status snapshot from {path}"
        ) from e
    if not isinstance(event, FullSnapshotEvent):
        raise CorruptStatusSnapshotError(
            f"Expected status snapshot at {path} to be a FullSnapshotEvent, "
            f"got {event.event_type}"
        )
    return event


def load_workflow_log_events(path: Path) -> list[RunModelWorkflowLogEvent]:
    """Read the workflow job invocations persisted for an experiment.

    An experiment that never ran a workflow has no such file, which is not an
    error, so a missing file yields no events.

    Lines that cannot be parsed are skipped rather than discarding the whole
    file: an interrupted ERT can leave a half-written final line behind, and
    the output that was written before that is still worth showing.

    Args:
        path: The ``workflow_events.jsonl`` file to read.

    Returns:
        One event per readable line, in the order they were written.

    Raises:
        OSError: If the file exists but cannot be read.
    """
    if not path.is_file():
        return []

    events = []
    for line_number, line in enumerate(
        path.read_text(encoding="utf-8").splitlines(), start=1
    ):
        if not line.strip():
            continue
        try:
            events.append(RunModelWorkflowLogEvent.model_validate_json(line))
        except ValueError:
            logger.warning(
                f"Skipping unreadable workflow event on line {line_number} of {path}",
                exc_info=True,
            )
    return events
