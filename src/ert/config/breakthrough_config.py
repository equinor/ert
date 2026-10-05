from __future__ import annotations

from datetime import datetime
from typing import Any, Literal

import polars as pl
from pydantic import Field

from .response_config import DerivedResponseConfig

_RESPONSE_SCHEMA: dict[str, Any] = {
    "response_key": pl.String,
    "threshold": pl.Float64,
    "time": pl.Datetime(time_unit="ms", time_zone=None),
    "values": pl.Float32,
}


class BreakthroughConfig(DerivedResponseConfig):
    type: Literal["breakthrough"] = "breakthrough"
    keys: list[str] = Field(default_factory=list)
    summary_keys: list[str] = Field(default_factory=list)
    thresholds: list[float] = Field(default_factory=list)
    observed_dates: list[datetime] = Field(default_factory=list)
    has_finalized_keys: bool = True

    @staticmethod
    def response_schema() -> dict[str, Any]:
        return _RESPONSE_SCHEMA

    def derive_from_storage(
        self, iter_: int, realization: int, ensemble: Any
    ) -> pl.DataFrame:
        breakthrough_times: list[datetime | None] = []
        breakthrough_time_offsets: list[float | None] = []
        for summary_key, threshold, obs_date in zip(
            self.summary_keys,
            self.thresholds,
            self.observed_dates,
            strict=True,
        ):
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
                offset_seconds = (breakthrough_time - obs_date).total_seconds()
                offset_days = offset_seconds / (60 * 60 * 24)
                breakthrough_time_offsets.append(offset_days)
                breakthrough_times.append(obs_date)

        if all(time is None for time in breakthrough_times):
            time_series = pl.Series(breakthrough_times, dtype=pl.Datetime)
            time_offset_series = pl.Series(
                breakthrough_time_offsets, dtype=_RESPONSE_SCHEMA["values"]
            )
        else:
            time_offset_series = pl.Series(
                breakthrough_time_offsets, dtype=_RESPONSE_SCHEMA["values"]
            )
            time_series = pl.Series(breakthrough_times)

        time_series = time_series.dt.cast_time_unit("ms")

        return pl.DataFrame(
            {
                "response_key": self.keys,
                "threshold": self.thresholds,
                "time": time_series,
                "values": time_offset_series,
            }
        ).pipe(self._assert_schema, self.response_schema())

    def fill_missing_values(
        self, response_df: pl.LazyFrame, ensemble_end_date: datetime | None
    ) -> pl.LazyFrame:
        """
        Fill in breakthrough responses whose threshold was not reached
        (null) within a realization's own simulated time range with the
        :term:`ensemble end date` + 1 day.
        """
        if not self.summary_keys or ensemble_end_date is None:
            return response_df

        collected = response_df.collect(engine="streaming")
        if collected.is_empty() or collected["values"].null_count() == 0:
            return collected.lazy()

        key_info = pl.DataFrame(
            {
                "response_key": self.keys,
                "threshold": self.thresholds,
                "obs_date": self.observed_dates,
            }
        )

        day_after_end_date = pl.lit(ensemble_end_date) + pl.duration(days=1)
        offset_days = (day_after_end_date - pl.col("obs_date")).dt.total_seconds() / (
            60 * 60 * 24
        )
        needs_fill = pl.col("values").is_null()

        return (
            collected.join(key_info, on=["response_key", *self.match_key], how="left")
            .with_columns(
                pl.when(needs_fill)
                .then(pl.col("obs_date"))
                .otherwise(pl.col("time"))
                .cast(_RESPONSE_SCHEMA["time"])
                .alias("time"),
                pl.when(needs_fill)
                .then(offset_days)
                .otherwise(pl.col("values"))
                .cast(_RESPONSE_SCHEMA["values"])
                .alias("values"),
            )
            .select(collected.columns)
            .lazy()
        )

    @property
    def match_key(self) -> list[str]:
        return ["threshold"]

    @classmethod
    def display_column(cls, value: Any, column_name: str) -> str:
        if column_name == "time":
            return value.strftime("%Y-%m-%d")

        return str(value)
