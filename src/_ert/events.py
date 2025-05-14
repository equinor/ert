from datetime import datetime
from typing import Annotated, Any, Final, Literal

from pydantic import BaseModel, ConfigDict, Field, TypeAdapter


class Id:
    FORWARD_MODEL_STEP_START_TYPE = Literal["forward_model_step.start"]
    FORWARD_MODEL_STEP_RUNNING_TYPE = Literal["forward_model_step.running"]
    FORWARD_MODEL_STEP_SUCCESS_TYPE = Literal["forward_model_step.success"]
    FORWARD_MODEL_STEP_FAILURE_TYPE = Literal["forward_model_step.failure"]
    FORWARD_MODEL_STEP_CHECKSUM_TYPE = Literal["forward_model_step.checksum"]
    FORWARD_MODEL_STEP_START: Final = "forward_model_step.start"
    FORWARD_MODEL_STEP_RUNNING: Final = "forward_model_step.running"
    FORWARD_MODEL_STEP_SUCCESS: Final = "forward_model_step.success"
    FORWARD_MODEL_STEP_FAILURE: Final = "forward_model_step.failure"
    FORWARD_MODEL_STEP_CHECKSUM: Final = "forward_model_step.checksum"

    REALIZATION_FAILURE_TYPE = Literal["realization.failure"]
    REALIZATION_PENDING_TYPE = Literal["realization.pending"]
    REALIZATION_RUNNING_TYPE = Literal["realization.running"]
    REALIZATION_SUCCESS_TYPE = Literal["realization.success"]
    REALIZATION_UNKNOWN_TYPE = Literal["realization.unknown"]
    REALIZATION_WAITING_TYPE = Literal["realization.waiting"]
    REALIZATION_TIMEOUT_TYPE = Literal["realization.timeout"]
    REALIZATION_STOPPED_LONG_RUNNING_TYPE = Literal["realization.stoppedlongrunning"]
    REALIZATION_RESUBMIT_TYPE = Literal["realization.resubmit"]
    REALIZATION_FAILURE: Final = "realization.failure"
    REALIZATION_PENDING: Final = "realization.pending"
    REALIZATION_RUNNING: Final = "realization.running"
    REALIZATION_SUCCESS: Final = "realization.success"
    REALIZATION_UNKNOWN: Final = "realization.unknown"
    REALIZATION_WAITING: Final = "realization.waiting"
    REALIZATION_TIMEOUT: Final = "realization.timeout"
    REALIZATION_STOPPED_LONG_RUNNING: Final = "realization.stoppedlongrunning"
    REALIZATION_RESUBMIT: Final = "realization.resubmit"

    ENSEMBLE_STARTED_TYPE = Literal["ensemble.started"]
    ENSEMBLE_SUCCEEDED_TYPE = Literal["ensemble.succeeded"]
    ENSEMBLE_CANCELLED_TYPE = Literal["ensemble.cancelled"]
    ENSEMBLE_FAILED_TYPE = Literal["ensemble.failed"]
    ENSEMBLE_STARTED: Final = "ensemble.started"
    ENSEMBLE_SUCCEEDED: Final = "ensemble.succeeded"
    ENSEMBLE_CANCELLED: Final = "ensemble.cancelled"
    ENSEMBLE_FAILED: Final = "ensemble.failed"
    ENSEMBLE_TYPES = (
        ENSEMBLE_STARTED_TYPE
        | ENSEMBLE_FAILED_TYPE
        | ENSEMBLE_SUCCEEDED_TYPE
        | ENSEMBLE_CANCELLED_TYPE
    )

    EE_SNAPSHOT_TYPE = Literal["ee.snapshot"]
    EE_SNAPSHOT_UPDATE_TYPE = Literal["ee.snapshot_update"]
    EE_SNAPSHOT: Final = "ee.snapshot"
    EE_SNAPSHOT_UPDATE: Final = "ee.snapshot_update"


class BaseEvent(BaseModel):
    model_config = ConfigDict(strict=True, extra="forbid")
    time: datetime = Field(default_factory=datetime.now)


class ForwardModelStepBaseEvent(BaseEvent):
    ensemble: str | None = None
    real: str
    fm_step: str


class ForwardModelStepStart(ForwardModelStepBaseEvent):
    event_type: Id.FORWARD_MODEL_STEP_START_TYPE = Id.FORWARD_MODEL_STEP_START
    std_out: str | None = None
    std_err: str | None = None


class ForwardModelStepRunning(ForwardModelStepBaseEvent):
    event_type: Id.FORWARD_MODEL_STEP_RUNNING_TYPE = Id.FORWARD_MODEL_STEP_RUNNING
    max_memory_usage: int | None = None
    current_memory_usage: int | None = None
    cpu_seconds: float = 0.0


class ForwardModelStepSuccess(ForwardModelStepBaseEvent):
    event_type: Id.FORWARD_MODEL_STEP_SUCCESS_TYPE = Id.FORWARD_MODEL_STEP_SUCCESS
    current_memory_usage: int | None = None


class ForwardModelStepFailure(ForwardModelStepBaseEvent):
    event_type: Id.FORWARD_MODEL_STEP_FAILURE_TYPE = Id.FORWARD_MODEL_STEP_FAILURE
    error_msg: str
    exit_code: int | None = None


class ForwardModelStepChecksum(BaseEvent):
    event_type: Id.FORWARD_MODEL_STEP_CHECKSUM_TYPE = Id.FORWARD_MODEL_STEP_CHECKSUM
    ensemble: str | None = None
    real: str
    checksums: dict[str, dict[str, Any]]


class RealizationBaseEvent(BaseEvent):
    real: str
    ensemble: str | None = None
    queue_event_type: str | None = None
    exec_hosts: str | None = None


class RealizationPending(RealizationBaseEvent):
    event_type: Id.REALIZATION_PENDING_TYPE = Id.REALIZATION_PENDING


class RealizationResubmit(RealizationBaseEvent):
    event_type: Id.REALIZATION_RESUBMIT_TYPE = Id.REALIZATION_RESUBMIT


class RealizationRunning(RealizationBaseEvent):
    event_type: Id.REALIZATION_RUNNING_TYPE = Id.REALIZATION_RUNNING


class RealizationSuccess(RealizationBaseEvent):
    event_type: Id.REALIZATION_SUCCESS_TYPE = Id.REALIZATION_SUCCESS


class RealizationFailed(RealizationBaseEvent):
    event_type: Id.REALIZATION_FAILURE_TYPE = Id.REALIZATION_FAILURE
    message: str | None = None  # Only used for JobState.FAILED


class RealizationUnknown(RealizationBaseEvent):
    event_type: Id.REALIZATION_UNKNOWN_TYPE = Id.REALIZATION_UNKNOWN


class RealizationWaiting(RealizationBaseEvent):
    event_type: Id.REALIZATION_WAITING_TYPE = Id.REALIZATION_WAITING


class RealizationTimeout(RealizationBaseEvent):
    event_type: Id.REALIZATION_TIMEOUT_TYPE = Id.REALIZATION_TIMEOUT


class RealizationStoppedLongRunning(RealizationBaseEvent):
    event_type: Id.REALIZATION_STOPPED_LONG_RUNNING_TYPE = (
        Id.REALIZATION_STOPPED_LONG_RUNNING
    )


class EnsembleBaseEvent(BaseEvent):
    ensemble: str | None = None


class EnsembleStarted(EnsembleBaseEvent):
    event_type: Id.ENSEMBLE_STARTED_TYPE = Id.ENSEMBLE_STARTED


class EnsembleSucceeded(EnsembleBaseEvent):
    event_type: Id.ENSEMBLE_SUCCEEDED_TYPE = Id.ENSEMBLE_SUCCEEDED


class EnsembleFailed(EnsembleBaseEvent):
    event_type: Id.ENSEMBLE_FAILED_TYPE = Id.ENSEMBLE_FAILED


class EnsembleCancelled(EnsembleBaseEvent):
    event_type: Id.ENSEMBLE_CANCELLED_TYPE = Id.ENSEMBLE_CANCELLED


class EESnapshot(EnsembleBaseEvent):
    event_type: Id.EE_SNAPSHOT_TYPE = Id.EE_SNAPSHOT
    snapshot: Any


class EESnapshotUpdate(EnsembleBaseEvent):
    event_type: Id.EE_SNAPSHOT_UPDATE_TYPE = Id.EE_SNAPSHOT_UPDATE
    snapshot: Any


FMEvent = (
    ForwardModelStepStart
    | ForwardModelStepRunning
    | ForwardModelStepSuccess
    | ForwardModelStepFailure
)

RealizationEvent = (
    RealizationPending
    | RealizationRunning
    | RealizationSuccess
    | RealizationFailed
    | RealizationTimeout
    | RealizationStoppedLongRunning
    | RealizationUnknown
    | RealizationWaiting
    | RealizationResubmit
)

EnsembleEvent = EnsembleStarted | EnsembleSucceeded | EnsembleFailed | EnsembleCancelled

EEEvent = EESnapshot | EESnapshotUpdate

SnapshotInputEvent = RealizationEvent | EnsembleEvent | FMEvent

DispatcherEvent = FMEvent | ForwardModelStepChecksum

_DISPATCH_EVENTS_ANNOTATION = Annotated[
    DispatcherEvent, Field(discriminator="event_type")
]
DispatcherEventAdapter: TypeAdapter[DispatcherEvent] = TypeAdapter(
    _DISPATCH_EVENTS_ANNOTATION
)


def dispatcher_event_from_json(raw_msg: str | bytes) -> DispatcherEvent:
    return DispatcherEventAdapter.validate_json(raw_msg)


def dispatcher_event_to_json(event: DispatcherEvent) -> str:
    return event.model_dump_json()
