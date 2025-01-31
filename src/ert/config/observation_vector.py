from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np

from .general_observation import GenObservation
from .parsing import ObservationType
from .summary_observation import SummaryObservation

if TYPE_CHECKING:
    from datetime import datetime

import polars as pl


@dataclass
class ObsVector:
    observation_type: ObservationType
    observation_key: str
    data_key: str
    observations: dict[int | datetime, GenObservation | SummaryObservation]

    def __iter__(self) -> Iterable[SummaryObservation | GenObservation]:
        """Iterate over active report steps; return node"""
        return iter(self.observations.values())

    def __len__(self) -> int:
        return len(self.observations)

    def to_dataset(self, active_list: list[int]) -> pl.DataFrame:
        if self.observation_type == ObservationType.GENERAL:
            dataframes = []
            for time_step, node in self.observations.items():
                if active_list and time_step not in active_list:
                    continue

                assert isinstance(node, GenObservation)
                dataframes.append(
                    pl.DataFrame(
                        {
                            "response_key": self.data_key,
                            "observation_key": self.observation_key,
                            "report_step": pl.Series(
                                np.full(len(node.indices), time_step),
                                dtype=pl.UInt16,
                            ),
                            "index": pl.Series(node.indices, dtype=pl.UInt16),
                            "observations": pl.Series(node.values, dtype=pl.Float32),
                            "std": pl.Series(node.stds, dtype=pl.Float32),
                        }
                    )
                )

            combined = pl.concat(dataframes)
            return combined
        elif self.observation_type == ObservationType.SUMMARY:
            observations = []
            actual_response_key = self.observation_key
            actual_observation_keys = []
            errors = []
            dates = list(self.observations.keys())
            if active_list:
                dates = [date for i, date in enumerate(dates) if i in active_list]

            for time_step in dates:
                n = self.observations[time_step]
                assert isinstance(n, SummaryObservation)
                actual_observation_keys.append(n.observation_key)
                observations.append(n.value)
                errors.append(n.std)

            dates_series = pl.Series(dates).dt.cast_time_unit("ms")

            return pl.DataFrame(
                {
                    "response_key": actual_response_key,
                    "observation_key": actual_observation_keys,
                    "time": dates_series,
                    "observations": pl.Series(observations, dtype=pl.Float32),
                    "std": pl.Series(errors, dtype=pl.Float32),
                }
            )
        else:
            raise ValueError(f"Unknown observation type {self.observation_type}")
