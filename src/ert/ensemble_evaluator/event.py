from dataclasses import dataclass
from typing import Dict, Literal, Optional

from .snapshot import Snapshot


@dataclass
class _UpdateEvent:
    iteration_label: str
    current_iteration: int
    total_iterations: int
    progress: float
    realization_count: int
    status_count: Dict[str, int]
    iteration: int


@dataclass
class FullSnapshotEvent(_UpdateEvent):
    event_type: Literal['FullSnapshotEvent']
    snapshot: Optional[Snapshot] = None


@dataclass
class SnapshotUpdateEvent(_UpdateEvent):
    event_type: Literal['SnapshotUpdateEvent']
    snapshot: Optional[Snapshot] = None


@dataclass
class EndEvent:
    failed: bool
    msg: Optional[str] = None
