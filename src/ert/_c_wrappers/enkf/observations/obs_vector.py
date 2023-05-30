from dataclasses import dataclass
from typing import TYPE_CHECKING, Dict, Iterable, List, Union

import numpy as np
import pandas as pd
from pandas import DataFrame, MultiIndex

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

    def get_gen_obs_data(self, active_list: List[int]) -> DataFrame:
        data = []
        data_key = self.data_key
        for time_step, node in self.observations.items():
            if active_list and time_step not in active_list:
                continue

            index = MultiIndex.from_product(
                [[self.observation_key], [f"{data_key}@{time_step}"], node.indices],
                names=["key", "data_key", "axis"],
            )
            data.append(
                DataFrame(
                    data=np.array([node.values, node.stds]).T,
                    index=index,
                    columns=["OBS", "STD"],
                )
            )
        return pd.concat(data)

    def get_summary_obs_data(self, obs: "EnkfObs", active_list: List[int]) -> DataFrame:
        observations = []

        active_steps = active_list if active_list else list(self.observations.keys())

        for time_step in active_steps:
            n: "SummaryObservation" = self.observations[time_step]  # type: ignore
            observations.append([n.value, n.std])
        data_key = self.data_key
        time_axis = [obs.obs_time[i] for i in active_steps]
        index = MultiIndex.from_product(
            [[self.observation_key], [data_key], time_axis],
            names=["key", "data_key", "axis"],
        )
        return DataFrame(
            data=np.array(observations), index=index, columns=["OBS", "STD"]
        )
