from __future__ import annotations

from typing import Literal

import polars as pl
from pydantic import Field

from ert.config.derived_response_config import DerivedResponseConfig


class BreakthroughConfig(DerivedResponseConfig):
    type: Literal["breakthrough"] = "breakthrough"
    keys: list[str] = Field(default_factory=list)
    thresholds: list[float] = Field(default_factory=list)
    has_finalized_keys: bool = True

    def to_derive(self, key: str, threshold: float) -> None:
        if key not in self.keys:
            self.keys.append(key)
            self.thresholds.append(threshold)

    @property
    def expected_input_files(self) -> list[str]:
        return []

    def derive_from_storage(self) -> pl.DataFrame:
        return pl.DataFrame()

    @property
    def primary_key(self) -> list[str]:
        return ["time"]

    @classmethod
    def from_config_dict(cls, config_dict: dict) -> BreakthroughConfig | None:
        return cls()
