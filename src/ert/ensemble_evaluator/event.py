from typing import Optional

from pydantic import BaseModel

from .snapshot import PartialSnapshot, Snapshot


class UpdateEvent(BaseModel):
    phase_name: str
    current_phase: int
    total_phases: int
    progress: float
    indeterminate: bool
    iteration: int


class FullSnapshotEvent(UpdateEvent):
    snapshot: Optional[Snapshot]

    class Config:
        arbitrary_types_allowed = True


class SnapshotUpdateEvent(UpdateEvent):
    partial_snapshot: Optional[PartialSnapshot]

    class Config:
        arbitrary_types_allowed = True


class EndEvent(BaseModel):
    failed: bool
    failed_msg: Optional[str]
