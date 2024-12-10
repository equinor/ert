from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from uuid import UUID

from ert.analysis.event import DataSection


@dataclass
class RunModelEvent:
    iteration: int
    run_id: UUID


@dataclass
class RunModelStatusEvent(RunModelEvent):
    msg: str


@dataclass
class RunModelTimeEvent(RunModelEvent):
    remaining_time: float
    elapsed_time: float


@dataclass
class RunModelUpdateBeginEvent(RunModelEvent):
    pass


@dataclass
class RunModelDataEvent(RunModelEvent):
    name: str
    data: DataSection

    def write_as_csv(self, output_path: Path | None) -> None:
        if output_path and self.data:
            self.data.to_csv(self.name, output_path / str(self.run_id))


@dataclass
class RunModelUpdateEndEvent(RunModelEvent):
    data: DataSection

    def write_as_csv(self, output_path: Path | None) -> None:
        if output_path and self.data:
            self.data.to_csv("Report", output_path / str(self.run_id))


@dataclass
class RunModelErrorEvent(RunModelEvent):
    error_msg: str
    data: DataSection | None = None

    def write_as_csv(self, output_path: Path | None) -> None:
        if output_path and self.data:
            self.data.to_csv("Report", output_path / str(self.run_id))
