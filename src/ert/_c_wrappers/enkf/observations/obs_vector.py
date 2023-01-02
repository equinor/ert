from typing import List, Union

from cwrap import BaseCClass

from ert import _clib
from ert._c_wrappers import ResPrototype
from ert._c_wrappers.enkf.config import EnkfConfigNode
from ert._c_wrappers.enkf.enums import EnkfObservationImplementationType
from ert._c_wrappers.enkf.observations.gen_observation import GenObservation
from ert._c_wrappers.enkf.observations.summary_observation import SummaryObservation


class ObsVector(BaseCClass):
    TYPE_NAME = "obs_vector"

    _alloc = ResPrototype(
        "void* obs_vector_alloc(enkf_obs_impl_type, char*, enkf_config_node, int)",
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
    _has_data = ResPrototype(
        "bool  obs_vector_has_data(obs_vector , bool_vector , enkf_fs)"
    )
    _get_config_node = ResPrototype(
        "enkf_config_node_ref obs_vector_get_config_node(obs_vector)"
    )
    _get_total_chi2 = ResPrototype(
        "double obs_vector_total_chi2(obs_vector, enkf_fs, int)"
    )
    _get_obs_key = ResPrototype("char*  obs_vector_get_obs_key(obs_vector)")

    def __init__(
        self,
        observation_type: EnkfObservationImplementationType,
        observation_key: str,
        config_node: EnkfConfigNode,
        num_reports: int,
    ):
        assert isinstance(observation_type, EnkfObservationImplementationType)
        assert isinstance(observation_key, str)
        assert isinstance(config_node, EnkfConfigNode)
        assert isinstance(num_reports, int)
        c_ptr = self._alloc(observation_type, observation_key, config_node, num_reports)
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

    def activeStep(self):
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

    def getConfigNode(self) -> EnkfConfigNode:
        return self._get_config_node().setParent(self)

    def hasData(self, active_mask, fs) -> bool:
        return self._has_data(active_mask, fs)

    def free(self):
        self._free()

    def __repr__(self):
        return (
            f"ObsVector(data_key = {self.getDataKey()}, "
            f"key = {self.getKey()}, obs_key = {self.getObsKey()}, "
            f"num_active = {len(self)}) {self._ad_str()}"
        )

    def getTotalChi2(self, fs, realization_number) -> float:
        return self._get_total_chi2(fs, realization_number)
