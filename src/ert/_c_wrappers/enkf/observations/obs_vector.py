from dataclasses import dataclass
from typing import TYPE_CHECKING, Dict, Iterable, List, Union

import xarray as xr

from ert._c_wrappers.enkf.enums import EnkfObservationImplementationType
from ert._c_wrappers.enkf.observations.gen_observation import GenObservation
from ert._c_wrappers.enkf.observations.summary_observation import SummaryObservation

if TYPE_CHECKING:
    from ert._c_wrappers.enkf.enkf_obs import EnkfObs


@dataclass
class ObsVector:
    observation_type: EnkfObservationImplementationType
    observation_key: str
    data_key: str
    observations: Dict[int, Union[GenObservation, SummaryObservation]]

    def __iter__(self) -> Iterable[Union[SummaryObservation, GenObservation]]:
        """Iterate over active report steps; return node"""
        return iter(self.observations.values())

    def __len__(self):
        return len(self.observations)

    def get_gen_obs_data(self, active_list: List[int]) -> xr.Dataset:
        datasets = []
        for time_step, node in self.observations.items():
            if active_list and time_step not in active_list:
                continue

            datasets.append(
                xr.Dataset(
                    {
                        "observations": (["report_step", "index"], [node.values]),
                        "std": (["report_step", "index"], [node.stds]),
                    },
                    coords={"index": node.indices, "report_step": [time_step]},
                )
            )
        return xr.combine_by_coords(datasets)

    def get_summary_obs_data(
        self, obs: "EnkfObs", active_list: List[int]
    ) -> xr.Dataset:
        observations = []
        errors = []
        active_steps = active_list if active_list else list(self.observations.keys())

        for time_step in active_steps:
            n: "SummaryObservation" = self.observations[time_step]
            observations.append(n.value)
            errors.append(n.std)
        data_key = self.data_key
        time_axis = [obs.obs_time[i] for i in active_steps]
        return xr.Dataset(
            {
                "observations": (["name", "time"], [observations]),
                "std": (["name", "time"], [errors]),
            },
            coords={"time": time_axis, "name": [data_key]},
        )
