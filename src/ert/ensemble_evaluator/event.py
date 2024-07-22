from dataclasses import dataclass
from typing import Dict, Optional

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
    snapshot: Optional[Snapshot] = None


@dataclass
class SnapshotUpdateEvent(_UpdateEvent):
    snapshot: Optional[Snapshot] = None


@dataclass
class EndEvent:
    failed: bool
    msg: Optional[str] = None
