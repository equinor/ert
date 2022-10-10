from typing import Iterator, List, Optional, Union

from cwrap import BaseCClass
from ecl.util.util import CTime, StringList

from ert._c_wrappers import ResPrototype
from ert._c_wrappers.enkf.enums import EnkfObservationImplementationType
from ert._c_wrappers.enkf.observations import ObsVector


class EnkfObs(BaseCClass):
    TYPE_NAME = "enkf_obs"

    _alloc = ResPrototype(
        "void* enkf_obs_alloc(history_source_enum, time_map, ecl_grid, \
                                        ecl_sum, ens_config)",
        bind=False,
    )
    _free = ResPrototype("void enkf_obs_free(enkf_obs)")
    _get_size = ResPrototype("int enkf_obs_get_size( enkf_obs )")
    _valid = ResPrototype("bool enkf_obs_is_valid(enkf_obs)")
    _load = ResPrototype("void enkf_obs_load(enkf_obs, char*, double)")
    _clear = ResPrototype("void enkf_obs_clear( enkf_obs )")
    _alloc_typed_keylist = ResPrototype(
        "stringlist_obj enkf_obs_alloc_typed_keylist(enkf_obs, enkf_obs_impl_type)"
    )
    _alloc_matching_keylist = ResPrototype(
        "stringlist_obj enkf_obs_alloc_matching_keylist(enkf_obs, char*)"
    )
    _has_key = ResPrototype("bool enkf_obs_has_key(enkf_obs, char*)")
    _obs_type = ResPrototype("enkf_obs_impl_type enkf_obs_get_type(enkf_obs, char*)")
    _get_vector = ResPrototype("obs_vector_ref enkf_obs_get_vector(enkf_obs, char*)")
    _iget_vector = ResPrototype("obs_vector_ref enkf_obs_iget_vector(enkf_obs, int)")
    _iget_obs_time = ResPrototype("time_t enkf_obs_iget_obs_time(enkf_obs, int)")
    _add_obs_vector = ResPrototype("void enkf_obs_add_obs_vector(enkf_obs, obs_vector)")

    def __init__(self, history_type, time_map, grid, refcase, ensemble_config):
        c_ptr = self._alloc(history_type, time_map, grid, refcase, ensemble_config)
        if c_ptr:
            super().__init__(c_ptr)
        else:
            raise ValueError("Failed to construct EnkfObs")

    def __len__(self):
        return self._get_size()

    def __contains__(self, key):
        return self._has_key(key)

    def __iter__(self) -> Iterator[ObsVector]:
        iobs = 0
        while iobs < len(self):
            vector = self[iobs]
            yield vector
            iobs += 1

    def __getitem__(self, key_or_index: Union[str, int]) -> ObsVector:
        if isinstance(key_or_index, str):
            if self.hasKey(key_or_index):
                return self._get_vector(key_or_index).setParent(self)
            else:
                raise KeyError(f"Unknown key: {key_or_index}")
        elif isinstance(key_or_index, int):
            idx = key_or_index
            if idx < 0:
                idx += len(self)
            if 0 <= idx < len(self):
                return self._iget_vector(idx).setParent(self)
            else:
                raise IndexError(
                    f"Invalid index: {key_or_index}.  Valid range is [0, {len(self)})."
                )
        else:
            raise TypeError(
                f"Key or index must be of type str or int, not {type(key_or_index)}."
            )

    def getTypedKeylist(
        self, observation_implementation_type: EnkfObservationImplementationType
    ) -> StringList:
        return self._alloc_typed_keylist(observation_implementation_type)

    def obsType(self, key):
        if key in self:
            return self._obs_type(key)
        else:
            raise KeyError(f"Unknown observation key: {key}")

    def getMatchingKeys(
        self, pattern: str, obs_type: Optional[EnkfObservationImplementationType] = None
    ) -> List[str]:
        """
        Will return a list of all the observation keys matching the input
        pattern. The matching is based on fnmatch().
        """
        key_list = self._alloc_matching_keylist(pattern)
        if obs_type:
            new_key_list = []
            for key in key_list:
                if self.obsType(key) == obs_type:
                    new_key_list.append(key)
            return new_key_list
        else:
            return key_list

    def hasKey(self, key) -> bool:
        return key in self

    def getObservationTime(self, index: int) -> CTime:
        return self._iget_obs_time(index)

    def addObservationVector(self, observation_vector):
        assert isinstance(observation_vector, ObsVector)

        observation_vector.convertToCReference(self)

        self._add_obs_vector(observation_vector)

    def free(self):
        self._free()

    def load(self, config_file, std_cutoff):
        self._load(config_file, std_cutoff)

    @property
    def valid(self):
        return self._valid()

    def clear(self):
        self._clear()

    def __repr__(self):
        validity = "valid" if self.valid else "invalid"
        return self._create_repr(f"{validity}, len={len(self)}")
