import os
from typing import TYPE_CHECKING, Iterator, List, Optional, Union

from cwrap import BaseCClass
from ecl.util.util import CTime, StringList

from ert import _clib
from ert._c_wrappers import ResPrototype
from ert._c_wrappers.enkf.enums import EnkfObservationImplementationType
from ert._c_wrappers.enkf.observations import ObsVector
from ert.parsing import ConfigValidationError

if TYPE_CHECKING:
    from ecl.summary import EclSum

    from ert._c_wrappers.enkf import EnsembleConfig, ErtConfig
    from ert._c_wrappers.enkf.time_map import TimeMap
    from ert._c_wrappers.sched import HistorySourceEnum


class ObservationConfigError(ConfigValidationError):
    @classmethod
    def _get_error_message(cls, config_file: Optional[str], error: str) -> str:
        return (
            (
                f"Parsing observations config file `{config_file}` "
                f"resulted in the errors: {error}"
            )
            if config_file is not None
            else error
        )


class EnkfObs(BaseCClass):
    TYPE_NAME = "enkf_obs"

    _free = ResPrototype("void enkf_obs_free(enkf_obs)")
    _get_size = ResPrototype("int enkf_obs_get_size( enkf_obs )")
    _error = ResPrototype("char* enkf_obs_get_error(enkf_obs)")
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

    def __init__(
        self,
        history_type: "HistorySourceEnum",
        time_map: Optional["TimeMap"],
        refcase: Optional["EclSum"],
        ensemble_config: "EnsembleConfig",
    ):
        # The c object holds on to ensemble_config, so we
        # need to hold onto a reference to it here so it does not
        # get destructed
        self._ensemble_config = ensemble_config
        c_ptr = _clib.enkf_obs.alloc(
            int(history_type), time_map, refcase, ensemble_config
        )
        if c_ptr:
            super().__init__(c_ptr)
        else:
            raise ValueError("Failed to construct EnkfObs")

    def __len__(self) -> int:
        return self._get_size()

    def __contains__(self, key: str) -> bool:
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

    def obsType(self, key: str) -> EnkfObservationImplementationType:
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
            return [key for key in key_list if self.obsType(key) == obs_type]
        else:
            return key_list

    def hasKey(self, key: str) -> bool:
        return key in self

    def getObservationTime(self, index: int) -> CTime:
        return self._iget_obs_time(index)

    def addObservationVector(self, observation_vector: ObsVector) -> None:
        assert isinstance(observation_vector, ObsVector)

        observation_vector.convertToCReference(self)

        self._add_obs_vector(observation_vector)

    def free(self):
        self._free()

    def load(self, config_file: str, std_cutoff: float) -> None:
        if not os.access(config_file, os.R_OK):
            raise RuntimeError(
                "Do not have permission to open observation "
                f"config file {config_file!r}"
            )
        _clib.enkf_obs.load(self, config_file, std_cutoff)

    @property
    def error(self) -> str:
        return self._error()

    def __repr__(self) -> str:
        return self._create_repr(f"{self.error}, len={len(self)}")

    @classmethod
    def from_ert_config(cls, config: "ErtConfig") -> "EnkfObs":
        ret = cls(
            config.model_config.history_source,
            config.model_config.time_map,
            config.ensemble_config.refcase,
            config.ensemble_config,
        )
        if config.model_config.obs_config_file:
            if (
                os.path.isfile(config.model_config.obs_config_file)
                and os.path.getsize(config.model_config.obs_config_file) == 0
            ):
                raise ObservationConfigError(
                    f"Empty observations file: "
                    f"{config.model_config.obs_config_file}"
                )

            if ret.error:
                raise ObservationConfigError(
                    f"Incorrect observations file: "
                    f"{config.model_config.obs_config_file}"
                    f": {ret.error}",
                    config_file=config.model_config.obs_config_file,
                )
            try:
                ret.load(
                    config.model_config.obs_config_file,
                    config.analysis_config.get_std_cutoff(),
                )
            except (ValueError, IndexError) as err:
                raise ObservationConfigError(
                    str(err),
                    config_file=config.model_config.obs_config_file,
                ) from err
        return ret
