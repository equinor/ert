import ctypes
import os.path

import numpy as np
from cwrap import BaseCClass
from ecl.util.util import IntVector

from ert import _clib
from ert._c_wrappers import ResPrototype
from ert._c_wrappers.enkf import ActiveList
from ert._c_wrappers.enkf.config import GenDataConfig


class GenObservation(BaseCClass):
    TYPE_NAME = "gen_obs"

    _alloc = ResPrototype("void*  gen_obs_alloc__(gen_data_config , char*)", bind=False)
    _free = ResPrototype("void   gen_obs_free(gen_obs)")
    _load = ResPrototype("void   gen_obs_load_observation(gen_obs , char*)")
    _scalar_set = ResPrototype("void   gen_obs_set_scalar(gen_obs , double , double)")
    _get_std = ResPrototype("double gen_obs_iget_std(gen_obs, int)")
    _get_value = ResPrototype("double gen_obs_iget_value(gen_obs, int)")
    _get_std_scaling = ResPrototype("double gen_obs_iget_std_scaling(gen_obs, int)")
    _get_size = ResPrototype("int    gen_obs_get_size(gen_obs)")
    _get_data_index = ResPrototype("int    gen_obs_get_obs_index(gen_obs, int)")
    _load_data_index = ResPrototype("void   gen_obs_load_data_index(gen_obs , char*)")
    _add_data_index = ResPrototype(
        "void   gen_obs_attach_data_index(gen_obs , int_vector)"
    )
    _get_value_vector = ResPrototype(
        "void   gen_obs_load_values(gen_obs, int, double*)"
    )
    _get_std_vector = ResPrototype("void   gen_obs_load_std(gen_obs, int, double*)")

    def __init__(
        self,
        obs_key,
        data_config: GenDataConfig,
        scalar_value=None,
        obs_file=None,
        data_index=None,
    ):
        c_ptr = self._alloc(data_config, obs_key)
        if c_ptr:
            super().__init__(c_ptr)
        else:
            raise ValueError(
                "Unable to construct GenObservation with given obs_key and data_config!"
            )

        if scalar_value is None and obs_file is None:
            raise ValueError(
                "Exactly one the scalar_value and obs_file arguments must be present"
            )

        if scalar_value is not None and obs_file is not None:
            raise ValueError(
                "Exactly one the scalar_value and obs_file arguments must be present"
            )

        if obs_file is not None:
            if not os.path.isfile(obs_file):
                raise IOError(
                    f"The file with observation data:{obs_file} does not exist"
                )
            self._load(obs_file)
        else:
            obs_value, obs_std = scalar_value
            self._scalar_set(obs_value, obs_std)

        if data_index is not None:
            if os.path.isfile(data_index):
                self._load_data_index(data_index)
            else:
                index_list = IntVector.active_list(data_index)
                self._add_data_index(index_list)

    def __len__(self):
        return self._get_size()

    def __getitem__(self, obs_index):
        if obs_index < 0:
            obs_index += len(self)

        if 0 <= obs_index < len(self):
            return (self.getValue(obs_index), self.getStandardDeviation(obs_index))
        else:
            raise IndexError(f"Invalid index.  Valid range: [0,{len(self)})")

    def getValue(self, obs_index: int) -> float:
        return self._get_value(obs_index)

    def getStandardDeviation(self, obs_index: int) -> float:
        return self._get_std(obs_index)

    def getStdScaling(self, obs_index: int) -> float:
        return self._get_std_scaling(obs_index)

    def updateStdScaling(self, factor, active_list: ActiveList):
        _clib.local.gen_obs.update_std_scaling(self, factor, active_list)

    def get_data_points(self):
        np_vector = np.zeros(len(self))
        self._get_value_vector(
            len(self), np_vector.ctypes.data_as(ctypes.POINTER(ctypes.c_double))
        )
        return np_vector

    def get_std(self):
        np_vector = np.zeros(len(self))
        self._get_std_vector(
            len(self), np_vector.ctypes.data_as(ctypes.POINTER(ctypes.c_double))
        )
        return np_vector

    def getSize(self) -> int:
        return len(self)

    def getIndex(self, obs_index) -> int:
        return self.getDataIndex(obs_index)

    def getDataIndex(self, obs_index):
        return self._get_data_index(obs_index)

    def free(self):
        self._free()

    def __repr__(self):
        return f"GenObservation(size = {len(self)}) {self._ad_str()}"
