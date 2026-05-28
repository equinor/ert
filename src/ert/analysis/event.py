from __future__ import annotations

import csv
import re
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field
from tabulate import tabulate


class AnalysisEvent(BaseModel):
    model_config = ConfigDict(arbitrary_types_allowed=True, extra="forbid")


class AnalysisStatusEvent(AnalysisEvent):
    event_type: Literal["AnalysisStatusEvent"] = "AnalysisStatusEvent"
    msg: str
    detail: bool = False


class AnalysisTimeEvent(AnalysisEvent):
    event_type: Literal["AnalysisTimeEvent"] = "AnalysisTimeEvent"
    remaining_time: float
    elapsed_time: float


class AnalysisReportEvent(AnalysisEvent):
    event_type: Literal["AnalysisReportEvent"] = "AnalysisReportEvent"
    report: str


@dataclass
class DataSection:
    header: list[str]
    data: Sequence[Sequence[str | float]]
    extra: dict[str, str] | None = None

    def __post_init__(self) -> None:
        if len(self.data) > 0 and len(self.header) != len(self.data[0]):
            raise ValueError(
                f"Header ({self.header}) must have same length as "
                f"number of columns ({len(self.data[0])})"
            )

    def to_csv(self, name: str, output_path: Path) -> None:
        fname = str(name).strip().replace(" ", "_")
        fname = re.sub(r"(?u)[^-\w]", "", fname)
        f_path = output_path / fname
        f_path.parent.mkdir(parents=True, exist_ok=True)
        with f_path.with_suffix(".report").open("w", encoding="utf-8") as fout:
            if self.extra:
                fout.writelines(f"{k}: {v}\n" for k, v in self.extra.items())
            fout.write(
                tabulate(
                    self.data,
                    headers=self.header,
                    tablefmt="simple_outline",
                    floatfmt=".4f",
                )
            )
        with f_path.with_suffix(".csv").open("w", encoding="utf-8", newline="") as fout:
            writer = csv.writer(fout, lineterminator="\n")
            writer.writerow(self.header)
            writer.writerows(self.data)


class AnalysisDataEvent(AnalysisEvent):
    event_type: Literal["AnalysisDataEvent"] = "AnalysisDataEvent"
    name: str
    data: DataSection


class AnalysisErrorEvent(AnalysisEvent):
    event_type: Literal["AnalysisErrorEvent"] = "AnalysisErrorEvent"
    error_msg: str
    data: DataSection | None


class AnalysisCompleteEvent(AnalysisEvent):
    event_type: Literal["AnalysisCompleteEvent"] = "AnalysisCompleteEvent"
    update_algorithm: str
    data: DataSection


class AnalysisMatrixEvent(AnalysisEvent):
    event_type: Literal["AnalysisMatrixEvent"] = "AnalysisMatrixEvent"
    name: str
    sparse: bool
    shape: tuple[int, int]
    data_type: str
    update_algorithm: str
    matrix_bytes: bytes = Field(exclude=True)
