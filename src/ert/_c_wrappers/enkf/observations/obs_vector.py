from datetime import datetime
from typing import TYPE_CHECKING, List, Union

import numpy as np
import pandas as pd
from cwrap import BaseCClass
from pandas import DataFrame, MultiIndex

from ert import _clib
from ert._c_wrappers import ResPrototype
from ert._c_wrappers.enkf.enums import EnkfObservationImplementationType
from ert._c_wrappers.enkf.observations.gen_observation import GenObservation
from ert._c_wrappers.enkf.observations.summary_observation import SummaryObservation

if TYPE_CHECKING:
    from ert._c_wrappers.enkf.enkf_obs import EnkfObs


class ObsVector(BaseCClass):
    TYPE_NAME = "obs_vector"

    _alloc = ResPrototype(
        "void* obs_vector_alloc(enkf_obs_impl_type, char*, char*, int)",
        bind=False,
    )
    _free = ResPrototype("void  obs_vector_free( obs_vector )")
    _get_state_kw = ResPrototype("char* obs_vector_get_state_kw( obs_vector )")
    _get_key = ResPrototype("char* obs_vector_get_key( obs_vector )")
    _iget_node = ResPrototype("void* obs_vector_iget_node( obs_vector, int)")
    _get_num_active = ResPrototype("int   obs_vector_get_num_active( obs_vector )")
    _iget_active = ResPrototype("bool  obs_vector_iget_active( obs_vector, int)")
    _get_impl_type = ResPrototype(
        "enkf_obs_impl_type obs_vector_get_impl_type( obs_vector)"
    )
    _install_node = ResPrototype(
        "void  obs_vector_install_node(obs_vector, int, void*)"
    )
    _get_next_active_step = ResPrototype(
        "int   obs_vector_get_next_active_step(obs_vector, int)"
    )
    _get_obs_key = ResPrototype("char*  obs_vector_get_obs_key(obs_vector)")

    def __init__(
        self,
        observation_type: EnkfObservationImplementationType,
        observation_key: str,
        config_node_key: str,
        num_reports: int,
    ):
        assert isinstance(observation_type, EnkfObservationImplementationType)
        assert isinstance(observation_key, str)
        assert isinstance(config_node_key, str)
        assert isinstance(num_reports, int)
        c_ptr = self._alloc(
            observation_type, observation_key, config_node_key, num_reports
        )
        super().__init__(c_ptr)

    def getDataKey(self) -> str:
        return self._get_state_kw()

    def getObservationKey(self) -> str:
        return self.getKey()

    def getKey(self) -> str:
        return self._get_key()

    def getObsKey(self) -> str:
        return self._get_obs_key()

    def getNode(self, index: int) -> Union[SummaryObservation, GenObservation]:
        pointer = self._iget_node(index)

        node_type = self.getImplementationType()
        if node_type == EnkfObservationImplementationType.SUMMARY_OBS:
            return SummaryObservation.createCReference(pointer, self)
        elif node_type == EnkfObservationImplementationType.GEN_OBS:
            return GenObservation.createCReference(pointer, self)
        else:
            raise AssertionError(f"Node type '{node_type}' currently not supported!")

    def __iter__(self):
        """Iterate over active report steps; return node"""
        for step in self.getStepList():
            yield self.getNode(step)

    def getStepList(self) -> List[int]:
        """
        Will return an IntVector with the active report steps.
        """
        return _clib.obs_vector_get_step_list(self)

    def add_summary_obs(self, summary_obs: SummaryObservation, index: int) -> None:
        summary_obs.convertToCReference(self)
        return _clib.obs_vector.add_summary_obs(self, summary_obs, index)

    def add_general_obs(self, gen_obs: GenObservation, index: int) -> None:
        gen_obs.convertToCReference(self)
        return _clib.obs_vector.add_general_obs(self, gen_obs, index)

    def activeStep(self) -> List[int]:
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
        return self._get_num_active()

    def isActive(self, index: int) -> bool:
        return self._iget_active(index)

    def getNextActiveStep(self, previous_step: int = -1) -> int:
        return self._get_next_active_step(previous_step)

    def getImplementationType(self) -> EnkfObservationImplementationType:
        return self._get_impl_type()

    def installNode(self, index, node):
        assert isinstance(node, SummaryObservation)
        node.convertToCReference(self)
        self._install_node(index, node.from_param(node))

    def free(self):
        self._free()

    def __repr__(self):
        return (
            f"ObsVector(data_key = {self.getDataKey()}, "
            f"key = {self.getKey()}, obs_key = {self.getObsKey()}, "
            f"num_active = {len(self)}) {self._ad_str()}"
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
            observations.append([n.getValue(), n.getStandardDeviation()])
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
