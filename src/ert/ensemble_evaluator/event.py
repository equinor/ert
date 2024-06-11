from dataclasses import dataclass
from typing import Dict, Optional

from .snapshot import PartialSnapshot, Snapshot


@dataclass
class _UpdateEvent:
    phase_name: str
    current_phase: int
    total_phases: int
    progress: float
    realization_count: int
    status_count: Dict[str, int]
    iteration: int


@dataclass
class FullSnapshotEvent(_UpdateEvent):
    snapshot: Optional[Snapshot] = None


@dataclass
class SnapshotUpdateEvent(_UpdateEvent):
    partial_snapshot: Optional[PartialSnapshot] = None


@dataclass
class EndEvent:
    failed: bool
    failed_msg: Optional[str] = None
