from __future__ import annotations

import re
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Union

import pandas as pd


@dataclass
class AnalysisEvent:
    pass


@dataclass
class AnalysisStatusEvent(AnalysisEvent):
    msg: str


@dataclass
class AnalysisTimeEvent(AnalysisEvent):
    remaining_time: float
    elapsed_time: float


@dataclass
class AnalysisReportEvent(AnalysisEvent):
    report: str


@dataclass
class DataSection:
    header: List[str]
    data: Sequence[Sequence[Union[str, float]]]
    extra: Optional[Dict[str, str]] = None

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
        df = pd.DataFrame(self.data, columns=self.header)
        with open(f_path.with_suffix(".report"), "w", encoding="utf-8") as fout:
            if self.extra:
                for k, v in self.extra.items():
                    fout.write(f"{k}: {v}\n")
            fout.write(df.to_markdown(tablefmt="simple_outline", floatfmt=".4f"))
        df.to_csv(f_path.with_suffix(".csv"))


@dataclass
class AnalysisDataEvent(AnalysisEvent):
    name: str
    data: DataSection


@dataclass
class AnalysisErrorEvent(AnalysisEvent):
    error_msg: str
    data: Optional[DataSection] = None


@dataclass
class AnalysisCompleteEvent(AnalysisEvent):
    data: DataSection
