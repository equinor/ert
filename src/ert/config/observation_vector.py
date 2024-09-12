from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Dict, Iterable, List, Union

import numpy as np

from .enkf_observation_implementation_type import EnkfObservationImplementationType
from .general_observation import GenObservation
from .summary_observation import SummaryObservation

if TYPE_CHECKING:
    from datetime import datetime

import polars


@dataclass
class ObsVector:
    observation_type: EnkfObservationImplementationType
    observation_key: str
    data_key: str
    observations: Dict[Union[int, datetime], Union[GenObservation, SummaryObservation]]

    def __iter__(self) -> Iterable[Union[SummaryObservation, GenObservation]]:
        """Iterate over active report steps; return node"""
        return iter(self.observations.values())

    def __len__(self) -> int:
        return len(self.observations)

    def to_dataset(self, active_list: List[int]) -> polars.DataFrame:
        if self.observation_type == EnkfObservationImplementationType.GEN_OBS:
            dataframes = []
            for time_step, node in self.observations.items():
                if active_list and time_step not in active_list:
                    continue

                assert isinstance(node, GenObservation)
                dataframes.append(
                    polars.DataFrame(
                        {
                            "response_key": self.data_key,
                            "observation_key": self.observation_key,
                            "report_step": polars.Series(
                                np.full(len(node.indices), time_step),
                                dtype=polars.UInt16,
                            ),
                            "index": polars.Series(node.indices, dtype=polars.UInt16),
                            "observations": polars.Series(
                                node.values, dtype=polars.Float32
                            ),
                            "std": polars.Series(node.stds, dtype=polars.Float32),
                        }
                    )
                )

            combined = polars.concat(dataframes)
            return combined
        elif self.observation_type == EnkfObservationImplementationType.SUMMARY_OBS:
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

            dates_series = polars.Series(dates).dt.cast_time_unit("ms")

            return polars.DataFrame(
                {
                    "response_key": actual_response_key,
                    "observation_key": actual_observation_keys,
                    "time": dates_series,
                    "observations": polars.Series(observations, dtype=polars.Float32),
                    "std": polars.Series(errors, dtype=polars.Float32),
                }
            )
        else:
            raise ValueError(f"Unknown observation type {self.observation_type}")
