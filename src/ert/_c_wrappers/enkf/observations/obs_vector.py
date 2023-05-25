from datetime import datetime
from typing import TYPE_CHECKING, Iterable, List, Optional, Union

import numpy as np
import pandas as pd
from pandas import DataFrame, MultiIndex
from sortedcontainers import SortedList

from ert._c_wrappers.enkf.enums import EnkfObservationImplementationType
from ert._c_wrappers.enkf.observations.gen_observation import GenObservation
from ert._c_wrappers.enkf.observations.summary_observation import SummaryObservation

if TYPE_CHECKING:
    from ert._c_wrappers.enkf.enkf_obs import EnkfObs


class ObsVector:
    def __init__(
        self,
        observation_type: EnkfObservationImplementationType,
        observation_key: str,
        config_node_key: str,
        num_steps: int,
    ):
        self.observation_type = observation_type
        self.observation_key = observation_key
        self.config_node_key = config_node_key
        self.nodes: List[Optional[Union[SummaryObservation, GenObservation]]] = [
            None
        ] * num_steps
        self._steps = SortedList()

    def getDataKey(self) -> str:
        return self.config_node_key

    def getObservationKey(self) -> str:
        return self.observation_key

    def getKey(self) -> str:
        return self.observation_key

    def getObsKey(self) -> str:
        return self.observation_key

    def getNode(
        self, index: int
    ) -> Optional[Union[SummaryObservation, GenObservation]]:
        return self.nodes[index]

    def __iter__(self) -> Iterable[Union[SummaryObservation, GenObservation]]:
        """Iterate over active report steps; return node"""
        return (self.nodes[i] for i in self.getStepList())  # type:ignore

    def getStepList(self) -> List[int]:
        """
        Will return an IntVector with the active report steps.
        """
        return list(self._steps)

    def add_summary_obs(self, summary_obs: SummaryObservation, index: int) -> None:
        self.nodes[index] = summary_obs
        self._steps.add(index)

    def add_general_obs(self, gen_obs: GenObservation, index: int) -> None:
        self.nodes[index] = gen_obs
        self._steps.add(index)

    def activeStep(self) -> int:
        """Assuming the observation is only active for one report step, this
        method will return that report step - if it is active for more
        than one report step the method will raise an exception.
        """
        step_list = self.getStepList()
        if len(step_list) == 1:
            return step_list[0]
        else:
            raise ValueError(
                "The activeStep() method can *ONLY* be called "
                "for obervations with one active step"
            )

    def firstActiveStep(self) -> int:
        step_list = self.getStepList()
        if len(step_list) > 0:
            return step_list[0]
        else:
            raise ValueError(
                "the firstActiveStep() method cannot be called with no active steps."
            )

    def getActiveCount(self) -> int:
        return len(self)

    def __len__(self):
        return len(self._steps)

    def isActive(self, index: int) -> bool:
        return index in self._steps

    def getImplementationType(self) -> EnkfObservationImplementationType:
        return self.observation_type

    def __repr__(self):
        return (
            f"ObsVector(observation_type={self.observation_type}, "
            f"observation_key={self.observation_key}, "
            f"config_node_key={self.config_node_key}, "
            f"num_steps = {len(self.nodes)})"
        )

    def get_gen_obs_data(self, active_list: List[int]) -> DataFrame:
        data = []
        data_key = self.getDataKey()
        for time_step in self.getStepList():
            if active_list and time_step not in active_list:
                continue
            node = self.getNode(time_step)

            index_list = [node.getIndex(nr) for nr in range(len(node))]
            index = MultiIndex.from_product(
                [[self.getKey()], [f"{data_key}@{time_step}"], index_list],
                names=["key", "data_key", "axis"],
            )
            values = node.get_data_points()
            errors = node.get_std()

            data.append(
                DataFrame(
                    data=np.array([values, errors]).T,
                    index=index,
                    columns=["OBS", "STD"],
                )
            )
        return pd.concat(data)

    def get_summary_obs_data(self, obs: "EnkfObs", active_list: List[int]) -> DataFrame:
        observations = []

        active_steps = active_list if active_list else self.getStepList()

        for time_step in active_steps:
            n: "SummaryObservation" = self.getNode(time_step)
            observations.append([n.value, n.std])
        data_key = self.getDataKey()
        time_axis = [
            datetime.strptime(str(obs.getObservationTime(i)), "%Y-%m-%d %H:%M:%S")
            for i in active_steps
        ]
        index = MultiIndex.from_product(
            [[self.getObsKey()], [data_key], time_axis],
            names=["key", "data_key", "axis"],
        )
        return DataFrame(
            data=np.array(observations), index=index, columns=["OBS", "STD"]
        )
