from collections.abc import Mapping
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, field_serializer, field_validator

from .snapshot import EnsembleSnapshot


class _UpdateEvent(BaseModel):
    model_config = ConfigDict(arbitrary_types_allowed=True, extra="forbid")
    iteration_label: str
    total_iterations: int
    progress: float
    realization_count: int
    status_count: dict[str, int]
    iteration: int
    snapshot: EnsembleSnapshot | None = None

    @field_serializer("snapshot")
    def serialize_snapshot(
        self, value: EnsembleSnapshot | None
    ) -> dict[str, Any] | None:
        if value is None:
            return None
        return value.to_dict()

    @field_validator("snapshot", mode="before")
    @classmethod
    def validate_snapshot(
        cls, value: EnsembleSnapshot | Mapping[Any, Any]
    ) -> EnsembleSnapshot:
        if isinstance(value, EnsembleSnapshot):
            return value
        return EnsembleSnapshot.from_nested_dict(value)


class FullSnapshotEvent(_UpdateEvent):
    event_type: Literal["FullSnapshotEvent"] = "FullSnapshotEvent"


class SnapshotUpdateEvent(_UpdateEvent):
    event_type: Literal["SnapshotUpdateEvent"] = "SnapshotUpdateEvent"


class EndEvent(BaseModel):
    model_config = ConfigDict(arbitrary_types_allowed=True, extra="forbid")
    event_type: Literal["EndEvent"] = "EndEvent"
    failed: bool
    msg: str


class WarningEvent(BaseModel):
    model_config = ConfigDict(arbitrary_types_allowed=True, extra="forbid")
    event_type: Literal["WarningEvent"] = "WarningEvent"
    msg: str
