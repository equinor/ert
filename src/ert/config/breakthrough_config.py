from __future__ import annotations

from datetime import datetime
from typing import Any, Literal

import polars as pl
from polars import Float32
from pydantic import Field

from ert.config.derived_response_config import DerivedResponseConfig


def datetime_hourly_difference(obs_date: datetime, response_date: datetime) -> float:
    seconds_per_hour = 3600
    return (response_date - obs_date).total_seconds() / seconds_per_hour


def key_to_summary_key(key: str) -> str:
    if key.startswith("BREAKTHROUGH:"):
        return key[len("BREAKTHROUGH:") :]
    return key


class BreakthroughConfig(DerivedResponseConfig):
    type: Literal["breakthrough"] = "breakthrough"
    keys: list[str] = Field(default_factory=list)
    thresholds: list[float] = Field(default_factory=list)
    observed_dates: list[datetime] = Field(default_factory=list)
    has_finalized_keys: bool = True

    def to_derive(self, key: str, threshold: float, obs_date: datetime) -> None:
        if key not in self.keys:
            self.keys.append(f"BREAKTHROUGH:{key}")
            self.thresholds.append(threshold)
            self.observed_dates.append(obs_date)

    @property
    def expected_input_files(self) -> list[str]:
        return []

    def derive_from_storage(
        self, iter_: int, realization: int, ensemble: Any
    ) -> pl.DataFrame:
        breakthrough_times: list[datetime | None] = []
        breakthrough_time_offsets: list[float | None] = []
        for key, threshold, obs_date in zip(
            self.keys, self.thresholds, self.observed_dates, strict=True
        ):
            summary_key = key_to_summary_key(key)
            response_df = ensemble.load_responses(summary_key, [realization])
            times = response_df["time"].to_list()
            values = response_df["values"].to_list()

            breakthrough_time = next(
                (
                    time
                    for time, value in zip(times, values, strict=True)
                    if value >= threshold
                ),
                None,
            )
            if breakthrough_time is None:
                breakthrough_time_offsets.append(None)
                breakthrough_times.append(None)
            else:
                breakthrough_time_offset = datetime_hourly_difference(
                    obs_date, breakthrough_time
                )
                breakthrough_time_offsets.append(breakthrough_time_offset)
                breakthrough_times.append(breakthrough_time)

        if all(time is None for time in breakthrough_times):
            time_series = pl.Series(breakthrough_times, dtype=pl.Datetime)
            time_offset_series = pl.Series(breakthrough_time_offsets, dtype=pl.Float32)
        else:
            time_offset_series = pl.Series(breakthrough_time_offsets, dtype=Float32)
            time_series = pl.Series(breakthrough_times).dt.cast_time_unit("ms")

        return pl.DataFrame(
            {
                "response_key": self.keys,
                "threshold": self.thresholds,
                "time": time_series,
                "values": time_offset_series,
            }
        )

    @property
    def primary_key(self) -> list[str]:
        return ["threshold"]

    @classmethod
    def from_config_dict(cls, config_dict: dict[str, Any]) -> BreakthroughConfig | None:
        obs_config = config_dict.get("OBS_CONFIG")
        if obs_config and any(
            obs.get("type") == "BREAKTHROUGH_OBSERVATION" for obs in obs_config[1]
        ):
            return cls()
        return None

    def display_column(self, value: Any, column_name: str) -> str:
        if column_name == "time":
            return value.strftime("%Y-%m-%d")

        return str(value)
