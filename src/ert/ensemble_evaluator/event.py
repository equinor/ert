import json
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


def snapshot_event_from_json(json_str: str) -> FullSnapshotEvent | SnapshotUpdateEvent:
    json_dict = json.loads(json_str)
    snapshot = EnsembleSnapshot.from_nested_dict(json_dict["snapshot"])
    json_dict["snapshot"] = snapshot
    match json_dict.pop("type"):
        case "FullSnapshotEvent":
            return FullSnapshotEvent(**json_dict)
        case "SnapshotUpdateEvent":
            return SnapshotUpdateEvent(**json_dict)
        case unknown:
            raise TypeError(f"Unknown snapshot update event type {unknown}")


def snapshot_event_to_json(event: FullSnapshotEvent | SnapshotUpdateEvent) -> str:
    assert event.snapshot is not None
    return json.dumps(
        {
            "iteration_label": event.iteration_label,
            "total_iterations": event.total_iterations,
            "progress": event.progress,
            "realization_count": event.realization_count,
            "status_count": event.status_count,
            "iteration": event.iteration,
            "snapshot": event.snapshot.to_dict(),
            "type": event.__class__.__name__,
        }
    )
