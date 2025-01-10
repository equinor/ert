from dataclasses import dataclass

from .snapshot import EnsembleSnapshot


@dataclass
class _UpdateEvent:
    iteration_label: str
    total_iterations: int
    progress: float
    realization_count: int
    status_count: dict[str, int]
    iteration: int


@dataclass
class FullSnapshotEvent(_UpdateEvent):
    snapshot: EnsembleSnapshot | None = None


@dataclass
class SnapshotUpdateEvent(_UpdateEvent):
    snapshot: EnsembleSnapshot | None = None


@dataclass
class EndEvent:
    failed: bool
    msg: str | None = None
