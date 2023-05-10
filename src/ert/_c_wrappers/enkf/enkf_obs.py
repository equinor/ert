import os
import warnings
from datetime import datetime, timedelta
from typing import TYPE_CHECKING, Iterator, List, Literal, Optional, Tuple, Union

import numpy as np
from cwrap import BaseCClass
from ecl.summary import EclSumVarType
from ecl.util.util import CTime, StringList

from ert import _clib
from ert._c_wrappers import ResPrototype
from ert._c_wrappers.enkf.enums import EnkfObservationImplementationType, ErtImplType
from ert._c_wrappers.enkf.observations import ObsVector
from ert._c_wrappers.enkf.observations.gen_observation import GenObservation
from ert._c_wrappers.enkf.observations.summary_observation import SummaryObservation
from ert._c_wrappers.sched import HistorySourceEnum
from ert.parsing import ConfigValidationError, ConfigWarning
from ert.parsing.error_info import ErrorInfo

if TYPE_CHECKING:
    import numpy.typing as npt
    from ecl.summary import EclSum

    from ert._c_wrappers.enkf import EnsembleConfig, ErtConfig
    from ert._c_wrappers.enkf.time_map import TimeMap

DEFAULT_TIME_DELTA = timedelta(seconds=30)


class ObservationConfigError(ConfigValidationError):
    @classmethod
    def get_value_error_message(cls, info: ErrorInfo) -> str:
        return (
            (
                f"Parsing observations config file `{info.filename}` "
                f"resulted in the following errors: {info.message}"
            )
            if info.filename is not None
            else info.message
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
        time_map: Optional["TimeMap"],
        refcase: Optional["EclSum"],
    ):
        # The c object holds on to ensemble_config, so we
        # need to hold onto a reference to it here so it does not
        # get destructed
        c_ptr = _clib.enkf_obs.alloc(time_map, refcase)
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

    @staticmethod
    def _handle_error_mode(
        values: "npt.ArrayLike",
        error: float,
        error_min: float,
        error_mode: Literal["ABS", "REL", "RELMIN"],
    ) -> "npt.NDArray":
        values = np.asarray(values)
        if error_mode == "ABS":
            return np.full(values.shape, error)
        elif error_mode == "REL":
            return np.abs(values) * error
        elif error_mode == "RELMIN":
            return np.maximum(np.abs(values) * error, np.full(values.shape, error_min))
        raise ValueError(f"Unknown error mode {error_mode}")

    def _handle_history_observation(
        self,
        ensemble_config: "EnsembleConfig",
        conf_instance,
        std_cutoff: float,
        history_type: Optional[HistorySourceEnum],
    ):
        sub_instances = conf_instance.get_sub_instances("HISTORY_OBSERVATION")

        if sub_instances == []:
            return

        refcase = ensemble_config.refcase
        if refcase is None:
            raise ObservationConfigError("REFCASE is required for HISTORY_OBSERVATION")
        if history_type is None:
            raise ValueError("Need a history type in order to use history observations")

        for instance in sub_instances:
            summary_key = instance.name
            time_len = len(_clib.enkf_obs.obs_time(self))
            ensemble_config.add_summary_full(summary_key, refcase)
            obs_vector = ObsVector(
                EnkfObservationImplementationType.SUMMARY_OBS,
                summary_key,
                ensemble_config.getNode(summary_key).getKey(),
                time_len,
            )
            error = float(instance.get_value("ERROR"))
            error_min = float(instance.get_value("ERROR_MIN"))
            error_mode = instance.get_value("ERROR_MODE")

            if history_type == HistorySourceEnum.REFCASE_HISTORY:
                var_type = refcase.var_type(summary_key)
                local_key = None
                if var_type in [
                    EclSumVarType.ECL_SMSPEC_WELL_VAR,
                    EclSumVarType.ECL_SMSPEC_GROUP_VAR,
                ]:
                    summary_node = refcase.smspec_node(summary_key)
                    local_key = summary_node.keyword + "H:" + summary_node.wgname
                elif var_type == EclSumVarType.ECL_SMSPEC_FIELD_VAR:
                    summary_node = refcase.smspec_node(summary_key)
                    local_key = summary_node.keyword + "H"
            else:
                local_key = summary_key
            if local_key is not None and local_key in refcase:
                valid, values = _clib.enkf_obs.read_from_refcase(refcase, local_key)
                std_dev = self._handle_error_mode(values, error, error_min, error_mode)
                for segment_instance in instance.get_sub_instances("SEGMENT"):
                    start = int(segment_instance.get_value("START"))
                    stop = int(segment_instance.get_value("STOP"))
                    if start < 0:
                        warnings.warn(
                            "Segment out of bounds. Truncating start of segment to 0.",
                            category=ConfigWarning,
                        )
                        start = 0
                    if stop >= time_len:
                        warnings.warn(
                            "Segment out of bounds. Truncating"
                            f" end of segment to {time_len - 1}.",
                            category=ConfigWarning,
                        )
                        stop = time_len - 1
                    if start > stop:
                        warnings.warn(
                            "Segment start after stop. Truncating"
                            f" end of segment to {start}.",
                            category=ConfigWarning,
                        )
                        stop = start
                    if np.size(std_dev[start:stop]) == 0:
                        warnings.warn(
                            f"Segment {segment_instance.name} does not"
                            " contain any time steps. The interval "
                            f"[{start}, {stop}) does not intersect with steps in the"
                            "time map.",
                            category=ConfigWarning,
                        )
                    std_dev[start:stop] = self._handle_error_mode(
                        values[start:stop],
                        float(segment_instance.get_value("ERROR")),
                        float(segment_instance.get_value("ERROR_MIN")),
                        segment_instance.get_value("ERROR_MODE"),
                    )
                for i, (good, error, value) in enumerate(zip(valid, std_dev, values)):
                    if good:
                        if error <= std_cutoff:
                            warnings.warn(
                                "Too small observation error in observation"
                                f" {summary_key}:{i} - ignored",
                                category=ConfigWarning,
                            )
                            continue
                        obs_vector.add_summary_obs(
                            SummaryObservation(summary_key, summary_key, value, error),
                            i,
                        )

            self.addObservationVector(obs_vector)

    @staticmethod
    def _get_time(conf_instance, start_time: datetime) -> Tuple[datetime, str]:
        if conf_instance.has_value("DATE"):
            date_str = conf_instance.get_value("DATE")
            try:
                return datetime.fromisoformat(date_str), f"DATE={date_str}"
            except ValueError:
                try:
                    date = datetime.strptime(date_str, "%d/%m/%Y")
                    warnings.warn(
                        f"Deprecated time format {date_str}."
                        " Please use ISO date format YYYY-MM-DD",
                        category=ConfigWarning,
                    )
                    return date, f"DATE={date_str}"
                except ValueError as err:
                    raise ValueError(
                        f"Unsupported date format {date_str}."
                        " Please use ISO date format"
                    ) from err

        if conf_instance.has_value("DAYS"):
            days = float(conf_instance.get_value("DAYS"))
            return start_time + timedelta(days=days), f"DAYS={days}"
        if conf_instance.has_value("HOURS"):
            hours = float(conf_instance.get_value("HOURS"))
            return start_time + timedelta(hours=hours), f"HOURS={hours}"
        raise ValueError("Missing time specifier")

    @staticmethod
    def _find_nearest(
        time_map: List[datetime],
        time: datetime,
        threshold: timedelta = DEFAULT_TIME_DELTA,
    ):
        nearest_index = -1
        nearest_diff = None
        for i, t in enumerate(time_map):
            diff = abs(time - t)
            if diff < threshold:
                if nearest_diff is None or nearest_diff > diff:
                    nearest_diff = diff
                    nearest_index = i
        if nearest_diff is None:
            raise IndexError(f"{time} is not in the time map")
        return nearest_index

    @staticmethod
    def _get_restart(conf_instance, time_map: List[datetime]) -> int:
        if conf_instance.has_value("RESTART"):
            return int(conf_instance.get_value("RESTART"))
        time, date_str = EnkfObs._get_time(conf_instance, time_map[0])
        try:
            return EnkfObs._find_nearest(time_map, time)
        except IndexError as err:
            raise IndexError(
                f"Could not find {time} ({date_str}) in "
                f"the time map for observation {conf_instance.name}"
            ) from err

    @staticmethod
    def _make_value_and_std_dev(conf_instance) -> Tuple[float, float]:
        value = float(conf_instance.get_value("VALUE"))
        return (
            value,
            float(
                EnkfObs._handle_error_mode(
                    np.array(value),
                    float(conf_instance.get_value("ERROR")),
                    float(conf_instance.get_value("ERROR_MIN")),
                    conf_instance.get_value("ERROR_MODE"),
                )
            ),
        )

    def _handle_summary_observation(
        self, ensemble_config: "EnsembleConfig", conf_instance
    ):
        time_map = [datetime.utcfromtimestamp(t) for t in _clib.enkf_obs.obs_time(self)]
        for instance in conf_instance.get_sub_instances("SUMMARY_OBSERVATION"):
            summary_key = instance.get_value("KEY")
            obs_key = instance.name
            refcase = ensemble_config.refcase
            ensemble_config.add_summary_full(summary_key, refcase)
            obs_vector = ObsVector(
                EnkfObservationImplementationType.SUMMARY_OBS,  # type: ignore
                obs_key,
                ensemble_config.getNode(summary_key).getKey(),
                len(time_map),
            )
            value, std_dev = self._make_value_and_std_dev(instance)
            try:
                restart = self._get_restart(instance, time_map)
            except ValueError as err:
                raise ValueError(
                    f"Problem with date in summary observation {obs_key}: " + str(err)
                ) from err

            if restart == 0:
                raise ValueError(
                    "It is unfortunately not possible to use summary "
                    "observations from the start of the simulation. "
                    f"Problem with observation {obs_key} at "
                    f"{self._get_time(instance, time_map[0])}"
                )
            obs_vector.add_summary_obs(
                SummaryObservation(summary_key, obs_key, value, std_dev), restart
            )
            self.addObservationVector(obs_vector)

    def _handle_general_observation(
        self, ensemble_config: "EnsembleConfig", conf_instance
    ):
        time_map = [datetime.utcfromtimestamp(t) for t in _clib.enkf_obs.obs_time(self)]
        for instance in conf_instance.get_sub_instances("GENERAL_OBSERVATION"):
            state_kw = instance.get_value("DATA")
            if not ensemble_config.hasNodeGenData(state_kw):
                warnings.warn(
                    f"Ensemble key {state_kw} does not exist"
                    f" - ignoring observation {instance.name}",
                    category=ConfigWarning,
                )
                continue
            config_node = ensemble_config.getNode(state_kw)
            obs_key = instance.name
            obs_vector = ObsVector(
                EnkfObservationImplementationType.GEN_OBS,  # type: ignore
                obs_key,
                config_node.getKey(),
                len(time_map),
            )
            try:
                restart = self._get_restart(instance, time_map)
            except ValueError as err:
                raise ValueError(
                    f"Problem with date in summary observation {obs_key}: " + str(err)
                ) from err
            if config_node.getImplementationType() != ErtImplType.GEN_DATA:
                warnings.warn(
                    f"{state_kw} has implementation type:"
                    f"'{config_node.getImplementationType()}' - "
                    f"expected:'GEN_DATA' in observation:{obs_key}."
                    "The observation will be ignored",
                    category=ConfigWarning,
                )
                continue
            if not config_node.hasReportStep(restart):
                warnings.warn(
                    f"The GEN_DATA node:{state_kw} is not configured "
                    f"to load from report step:{restart} for the observation:{obs_key}"
                    " - The observation will be ignored",
                    category=ConfigWarning,
                )
                continue

            obs_vector.add_general_obs(
                GenObservation(
                    obs_key,
                    config_node,
                    (
                        float(instance.get_value("VALUE")),
                        float(instance.get_value("ERROR")),
                    )
                    if instance.has_value("VALUE")
                    else None,
                    instance.get_value("OBS_FILE")
                    if instance.has_value("OBS_FILE")
                    else None,
                    instance.get_value("INDEX_LIST")
                    if instance.has_value("INDEX_LIST")
                    else None,
                ),
                restart,
            )

            self.addObservationVector(obs_vector)

    def load(
        self,
        history: Optional[HistorySourceEnum],
        conf_instance: _clib.enkf_obs.ConfInstance,
        std_cutoff: float,
        ensemble_config: "EnsembleConfig",
    ) -> None:
        self._handle_history_observation(
            ensemble_config, conf_instance, std_cutoff, history
        )
        self._handle_summary_observation(ensemble_config, conf_instance)
        self._handle_general_observation(ensemble_config, conf_instance)

        obs_time_map = _clib.enkf_obs.get_time_map(self)

        for state_kw in set(obs_time_map.values()):
            obs_keys = [k for k, v in obs_time_map.items() if v == state_kw]
            obs_keys.sort()
            node = ensemble_config.getNode(state_kw)
            if node is not None:
                _clib.enkf_config_node.update_obs_keys(node, obs_keys)

    @property
    def error(self) -> str:
        return self._error()

    def __repr__(self) -> str:
        return self._create_repr(f"{self.error}, len={len(self)}")

    @classmethod
    def from_ert_config(cls, config: "ErtConfig") -> "EnkfObs":
        ret = cls(
            config.model_config.time_map,
            config.ensemble_config.refcase,
        )
        obs_config_file = config.model_config.obs_config_file
        if obs_config_file:
            if (
                os.path.isfile(obs_config_file)
                and os.path.getsize(obs_config_file) == 0
            ):
                raise ObservationConfigError(
                    [
                        ErrorInfo(
                            message=f"Empty observations file: {obs_config_file}",
                            filename=config.user_config_file,
                        ).set_context(obs_config_file)
                    ]
                )

            if ret.error:
                raise ObservationConfigError(
                    f"{ret.error}",
                    config_file=obs_config_file,
                )
            if not os.access(obs_config_file, os.R_OK):
                raise ObservationConfigError(
                    [
                        ErrorInfo(
                            message="Do not have permission to open observation"
                            f" config file {obs_config_file!r}",
                            filename=config.user_config_file,
                        ).set_context(obs_config_file)
                    ]
                )
            conf_instance = _clib.enkf_obs.ConfInstance(obs_config_file)
            errors = conf_instance.get_errors()
            if errors != []:
                raise ObservationConfigError(
                    errors=[
                        ErrorInfo(filename=obs_config_file, message=e) for e in errors
                    ]
                )
            try:
                ret.load(
                    config.model_config.history_source,
                    conf_instance,
                    config.analysis_config.get_std_cutoff(),
                    config.ensemble_config,
                )
            except IndexError as err:
                if config.ensemble_config.refcase is not None:
                    raise ObservationConfigError(
                        f"{err}. The time map is set from the REFCASE keyword. Either "
                        "the REFCASE has an incorrect/missing date, or the observation "
                        "is given an incorrect date.",
                        config_file=obs_config_file,
                    ) from err
                raise ObservationConfigError(
                    f"{err}. The time map is set from the TIME_MAP"
                    "keyword. Either the time map file has an"
                    "incorrect/missing date, or the  observation is given an"
                    "incorrect date.",
                    config_file=obs_config_file,
                ) from err

            except ValueError as err:
                raise ObservationConfigError(
                    str(err),
                    config_file=obs_config_file,
                ) from err
        return ret
