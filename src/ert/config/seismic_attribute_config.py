from dataclasses import field
from datetime import datetime
from pathlib import Path
from typing import Literal, Self

import polars as pl
from pydantic import BaseModel

from .parsing import ConfigDict
from .response_config import ResponseConfig


class _SeismicAttribute(BaseModel):
    start: datetime
    end: datetime
    file: Path


class SeismicAttributeConfig(ResponseConfig):
    type: Literal["seismic_attribute"] = "seismic_attribute"
    seismic_attributes: list[_SeismicAttribute] = field(default_factory=list)

    def read_from_file(self, run_path: str, iens: int, iter_: int) -> pl.DataFrame:
        return pl.DataFrame()

    @property
    def expected_input_files(self) -> list[str]:
        return []

    @property
    def primary_key(self) -> list[str]:
        return []

    @classmethod
    def from_config_dict(cls, config_dict: ConfigDict) -> Self | None:
        seismic_attributes = []
        for start, end, file in config_dict.get("SEISMIC_4D_ATTRIBUTE", []):
            seismic_attributes.append(
                _SeismicAttribute(
                    start=datetime.fromisoformat(start.replace("START:", "")),
                    end=datetime.fromisoformat(end.replace("END:", "")),
                    file=Path(file.replace("FILE:", "")),
                )
            )
        return (
            cls(
                seismic_attributes=seismic_attributes,
                keys=["foo"],  # To be implemented alongside observation type
            )
            if seismic_attributes
            else None
        )
