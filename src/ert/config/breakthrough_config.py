from __future__ import annotations

from typing import Any, Literal

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

    def derive_from_storage(
        self, iter: int, realization: int, ensemble: Any
    ) -> pl.DataFrame:
        breakthrough_times = []
        for key, threshold in zip(self.keys, self.thresholds, strict=True):
            response_df = ensemble.load_responses(key, [realization])
            times = response_df["time"].to_list()
            values = response_df["values"].to_list()

            breakthrough_times.append(
                next(
                    (
                        time
                        for time, value in zip(times, values, strict=True)
                        if value > threshold
                    ),
                    None,
                )
            )
        print(breakthrough_times)

        primary_keys = [
            f"{key}:{threshold}" for key, threshold in zip(self.keys, self.thresholds)
        ]
        return pl.DataFrame(
            {
                "response_key": primary_keys,
                "values": breakthrough_times,
            }
        )

    @property
    def primary_key(self) -> list[str]:
        return ["time"]

    @classmethod
    def from_config_dict(cls, config_dict: dict) -> BreakthroughConfig | None:
        return cls()
