from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Dict, Iterable, List, Union

import xarray as xr

from .enkf_observation_implementation_type import EnkfObservationImplementationType
from .general_observation import GenObservation
from .summary_observation import SummaryObservation

if TYPE_CHECKING:
    from datetime import datetime


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

    def to_dataset(self, active_list: List[int]) -> xr.Dataset:
        if self.observation_type == EnkfObservationImplementationType.GEN_OBS:
            datasets = []
            for time_step, node in self.observations.items():
                if active_list and time_step not in active_list:
                    continue

                assert isinstance(node, GenObservation)
                datasets.append(
                    xr.Dataset(
                        {
                            "observations": (["report_step", "index"], [node.values]),
                            "std": (["report_step", "index"], [node.stds]),
                        },
                        coords={"index": node.indices, "report_step": [time_step]},
                    )
                )
            combined = xr.combine_by_coords(datasets)
            combined.attrs["response"] = self.data_key
            return combined  # type: ignore
        elif self.observation_type == EnkfObservationImplementationType.SUMMARY_OBS:
            observations = []
            errors = []
            dates = list(self.observations.keys())
            if active_list:
                dates = [date for i, date in enumerate(dates) if i in active_list]

            for time_step in dates:
                n = self.observations[time_step]
                assert isinstance(n, SummaryObservation)
                observations.append(n.value)
                errors.append(n.std)
            return xr.Dataset(
                {
                    "observations": (["name", "time"], [observations]),
                    "std": (["name", "time"], [errors]),
                },
                coords={"time": dates, "name": [self.observation_key]},
                attrs={"response": "summary"},
            )
        else:
            raise ValueError(f"Unknown observation type {self.observation_type}")
