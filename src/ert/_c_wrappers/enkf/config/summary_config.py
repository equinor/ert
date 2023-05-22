from typing import List

from ert._c_wrappers.enkf.enums.enkf_var_type_enum import EnkfVarType
from ert._c_wrappers.enkf.enums.ert_impl_type_enum import ErtImplType


class SummaryConfig:
    def __init__(self, key):
        self.name = key
        self._observation_list: List[str] = []

    @property
    def var_type(self):
        return EnkfVarType.DYNAMIC_RESULT

    def update_observation_keys(self, observations: List[str]):
        self._observation_list = observations
        self._observation_list.sort()

    def get_observation_keys(self) -> List[str]:
        return self._observation_list

    def getImplementationType(self) -> ErtImplType:
        return ErtImplType.SUMMARY

    def getKey(self):
        return self.name

    def __repr__(self):
        return (
            f"SummaryConfig(key={self.name}, "
            f"observation_keys={self._observation_list})"
        )
