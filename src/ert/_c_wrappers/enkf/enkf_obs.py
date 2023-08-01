import os
import warnings
from datetime import datetime, timedelta
from fnmatch import fnmatch
from typing import TYPE_CHECKING, Dict, Iterator, List, Literal, Optional, Tuple, Union

import numpy as np
import xarray as xr
from ecl.summary import EclSumVarType
from ecl.util.util import CTime, IntVector

from ert._c_wrappers.enkf.enums import EnkfObservationImplementationType
from ert._c_wrappers.enkf.observations import ObsVector
from ert._c_wrappers.enkf.observations.gen_observation import GenObservation
from ert._c_wrappers.enkf.observations.summary_observation import SummaryObservation
from ert._c_wrappers.sched import HistorySource
from ert._clib.enkf_obs import read_from_refcase
from ert.config import GenDataConfig, SummaryConfig
from ert.parsing import ConfigWarning, ErrorInfo
from ert.parsing.new_observations_parser import (
    DateDict,
    GenObsValues,
    HistoryValues,
    ObservationConfigError,
    ObservationType,
    SummaryValues,
    parse,
)

if TYPE_CHECKING:
    import numpy.typing as npt

    from ert._c_wrappers.enkf import EnsembleConfig, ErtConfig

DEFAULT_TIME_DELTA = timedelta(seconds=30)


class EnkfObs:
    def __init__(self, obs_vectors: Dict[str, ObsVector], obs_time: List[datetime]):
        self.obs_vectors = obs_vectors
        self.obs_time = obs_time

    def __len__(self) -> int:
        return len(self.obs_vectors)

    def __contains__(self, key: str) -> bool:
        return key in self.obs_vectors

    def __iter__(self) -> Iterator[ObsVector]:
        return iter(self.obs_vectors.values())

    def __getitem__(self, key: str) -> ObsVector:
        return self.obs_vectors[key]

    def getTypedKeylist(
        self, observation_implementation_type: EnkfObservationImplementationType
    ) -> List[str]:
        return sorted(
            [
                key
                for key, obs in self.obs_vectors.items()
                if observation_implementation_type == obs.observation_type
            ]
        )

    def getMatchingKeys(
        self, pattern: str, obs_type: Optional[EnkfObservationImplementationType] = None
    ) -> List[str]:
        """
        Will return a list of all the observation keys matching the input
        pattern. The matching is based on fnmatch().
        """
        key_list = sorted(
            key
            for key in self.obs_vectors
            if any(fnmatch(key, p) for p in pattern.split())
        )
        if obs_type:
            return [
                key
                for key in key_list
                if self.obs_vectors[key].observation_type == obs_type
            ]
        else:
            return key_list

    def hasKey(self, key: str) -> bool:
        return key in self

    def getObservationTime(self, index: int) -> datetime:
        return self.obs_time[index]

    @staticmethod
    def _handle_error_mode(
        values: "npt.ArrayLike",
        error: float,
        error_min: float,
        error_mode: Literal["ABS", "REL", "RELMIN"],
    ) -> "npt.NDArray[np.double]":
        values = np.asarray(values)
        if error_mode == "ABS":
            return np.full(values.shape, error)
        elif error_mode == "REL":
            return np.abs(values) * error  # type: ignore
        elif error_mode == "RELMIN":
            return np.maximum(np.abs(values) * error, np.full(values.shape, error_min))
        raise ValueError(f"Unknown error mode {error_mode}")

    @classmethod
    def _handle_history_observation(
        cls,
        ensemble_config: "EnsembleConfig",
        history_observation: HistoryValues,
        summary_key: str,
        std_cutoff: float,
        history_type: Optional[HistorySource],
        time_len: int,
    ) -> Dict[str, ObsVector]:
        response_config = ensemble_config["summary"]
        assert isinstance(response_config, SummaryConfig)

        refcase = response_config.refcase
        if refcase is None:
            raise ObservationConfigError("REFCASE is required for HISTORY_OBSERVATION")
        if history_type is None:
            raise ValueError("Need a history type in order to use history observations")

        response_config.keys.append(summary_key)
        error = history_observation["ERROR"]
        error_min = history_observation["ERROR_MIN"]
        error_mode = history_observation["ERROR_MODE"]

        if history_type == HistorySource.REFCASE_HISTORY:
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
        if local_key is None:
            return {}
        if local_key not in refcase:
            return {}
        valid, values = read_from_refcase(refcase, local_key)
        std_dev = cls._handle_error_mode(values, error, error_min, error_mode)
        for segment_name, segment_instance in history_observation["SEGMENT"]:
            start = segment_instance["START"]
            stop = segment_instance["STOP"]
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
                    f"Segment {segment_name} does not"
                    " contain any time steps. The interval "
                    f"[{start}, {stop}) does not intersect with steps in the"
                    "time map.",
                    category=ConfigWarning,
                )
            std_dev[start:stop] = cls._handle_error_mode(
                values[start:stop],
                segment_instance["ERROR"],
                segment_instance["ERROR_MIN"],
                segment_instance["ERROR_MODE"],
            )
        data: Dict[int, Union[GenObservation, SummaryObservation]] = {}
        for i, (good, error, value) in enumerate(zip(valid, std_dev, values)):
            if good:
                if error <= std_cutoff:
                    warnings.warn(
                        "Too small observation error in observation"
                        f" {summary_key}:{i} - ignored",
                        category=ConfigWarning,
                    )
                    continue
                data[i] = SummaryObservation(summary_key, summary_key, value, error)

        return {
            summary_key: ObsVector(
                EnkfObservationImplementationType.SUMMARY_OBS,
                summary_key,
                "summary",
                data,
            )
        }

    @staticmethod
    def _get_time(date_dict: DateDict, start_time: datetime) -> Tuple[datetime, str]:
        if "DATE" in date_dict:
            date_str = date_dict["DATE"]
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

        if "DAYS" in date_dict:
            days = date_dict["DAYS"]
            return start_time + timedelta(days=days), f"DAYS={days}"
        if "HOURS" in date_dict:
            hours = date_dict["HOURS"]
            return start_time + timedelta(hours=hours), f"HOURS={hours}"
        raise ValueError("Missing time specifier")

    @staticmethod
    def _find_nearest(
        time_map: List[datetime],
        time: datetime,
        threshold: timedelta = DEFAULT_TIME_DELTA,
    ) -> int:
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
    def _get_restart(
        date_dict: DateDict, obs_name: str, time_map: List[datetime]
    ) -> int:
        if "RESTART" in date_dict:
            return date_dict["RESTART"]
        time, date_str = EnkfObs._get_time(date_dict, time_map[0])
        try:
            return EnkfObs._find_nearest(time_map, time)
        except IndexError as err:
            raise IndexError(
                f"Could not find {time} ({date_str}) in "
                f"the time map for observation {obs_name}"
            ) from err

    @staticmethod
    def _make_value_and_std_dev(
        observation_dict: SummaryValues,
    ) -> Tuple[float, float]:
        value = observation_dict["VALUE"]
        return (
            value,
            float(
                EnkfObs._handle_error_mode(
                    np.array(value),
                    observation_dict["ERROR"],
                    observation_dict["ERROR_MIN"],
                    observation_dict["ERROR_MODE"],
                )
            ),
        )

    @classmethod
    def _handle_summary_observation(
        cls,
        ensemble_config: "EnsembleConfig",
        summary_dict: SummaryValues,
        obs_key: str,
        time_map: List[datetime],
    ) -> Dict[str, ObsVector]:
        summary_key = summary_dict["KEY"]
        ensemble_config["summary"].keys.append(summary_key)  # type: ignore
        value, std_dev = cls._make_value_and_std_dev(summary_dict)
        try:
            restart = cls._get_restart(summary_dict, obs_key, time_map)
        except ValueError as err:
            raise ValueError(
                f"Problem with date in summary observation {obs_key}: " + str(err)
            ) from err

        if restart == 0:
            raise ValueError(
                "It is unfortunately not possible to use summary "
                "observations from the start of the simulation. "
                f"Problem with observation {obs_key} at "
                f"{cls._get_time(summary_dict, time_map[0])}"
            )
        return {
            obs_key: ObsVector(
                EnkfObservationImplementationType.SUMMARY_OBS,
                summary_key,
                "summary",
                {restart: SummaryObservation(summary_key, obs_key, value, std_dev)},
            )
        }

    @classmethod
    def _create_gen_obs(
        cls,
        scalar_value: Optional[Tuple[float, float]] = None,
        obs_file: Optional[str] = None,
        data_index: Optional[str] = None,
    ) -> GenObservation:
        if scalar_value is None and obs_file is None:
            raise ValueError(
                "Exactly one the scalar_value and obs_file arguments must be present"
            )

        if scalar_value is not None and obs_file is not None:
            raise ValueError(
                "Exactly one the scalar_value and obs_file arguments must be present"
            )

        if obs_file is not None:
            file_values = np.loadtxt(obs_file, delimiter=None).ravel()
            if len(file_values) % 2 != 0:
                raise ValueError(
                    "Expected even number of values in GENERAL_OBSERVATION"
                )
            values = file_values[::2]
            stds = file_values[1::2]

        else:
            assert scalar_value is not None
            obs_value, obs_std = scalar_value
            values = np.array([obs_value])
            stds = np.array([obs_std])

        if data_index is not None:
            indices = np.array([])
            if os.path.isfile(data_index):
                indices = np.loadtxt(data_index, delimiter=None).ravel()
            else:
                indices = np.array(IntVector.active_list(data_index), dtype=np.int32)
        else:
            indices = np.arange(len(values))
        std_scaling = np.full(len(values), 1.0)
        return GenObservation(values, stds, indices, std_scaling)

    @classmethod
    def _handle_general_observation(
        cls,
        ensemble_config: "EnsembleConfig",
        general_observation: GenObsValues,
        obs_key: str,
        time_map: List[datetime],
    ) -> Dict[str, ObsVector]:
        state_kw = general_observation["DATA"]
        if not ensemble_config.hasNodeGenData(state_kw):
            warnings.warn(
                f"Ensemble key {state_kw} does not exist"
                f" - ignoring observation {obs_key}",
                category=ConfigWarning,
            )
            return {}
        config_node = ensemble_config.getNode(state_kw)
        try:
            restart = cls._get_restart(general_observation, obs_key, time_map)
        except ValueError as err:
            raise ValueError(
                f"Problem with date in summary observation {obs_key}: " + str(err)
            ) from err
        if not isinstance(config_node, GenDataConfig):
            warnings.warn(
                f"{state_kw} has implementation type:"
                f"'{type(config_node)}' - "
                f"expected:'GEN_DATA' in observation:{obs_key}."
                "The observation will be ignored",
                category=ConfigWarning,
            )
            return {}
        if restart not in config_node.report_steps:
            warnings.warn(
                f"The GEN_DATA node:{state_kw} is not configured "
                f"to load from report step:{restart} for the observation:{obs_key}"
                " - The observation will be ignored",
                category=ConfigWarning,
            )
            return {}

        return {
            obs_key: ObsVector(
                EnkfObservationImplementationType.GEN_OBS,
                obs_key,
                config_node.name,
                {
                    restart: cls._create_gen_obs(
                        (
                            general_observation["VALUE"],
                            general_observation["ERROR"],
                        )
                        if "VALUE" in general_observation
                        else None,
                        general_observation["OBS_FILE"]
                        if "OBS_FILE" in general_observation
                        else None,
                        general_observation["INDEX_LIST"]
                        if "INDEX_LIST" in general_observation
                        else None,
                    ),
                },
            )
        }

    def __repr__(self) -> str:
        return f"EnkfObs({self.obs_vectors}, {self.obs_time})"

    @classmethod
    def from_ert_config(  # pylint: disable=too-many-branches
        cls,
        config: "ErtConfig",
    ) -> "EnkfObs":
        obs_config_file = config.model_config.obs_config_file
        obs_time_list: List[datetime] = []
        if config.model_config.refcase is not None:
            refcase = config.model_config.refcase
            obs_time_list = [refcase.get_start_time()] + [
                CTime(t).datetime() for t in refcase.alloc_time_vector(True)
            ]
        elif config.model_config.time_map is not None:
            time_map = config.model_config.time_map
            obs_time_list = [time_map[i] for i in range(len(time_map))]
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
            if obs_time_list == []:
                raise ObservationConfigError("Missing refcase or TIMEMAP")
            obs_config_content = parse(obs_config_file)
            try:
                history = config.model_config.history_source
                std_cutoff = config.analysis_config.get_std_cutoff()
                time_len = len(obs_time_list)
                ensemble_config = config.ensemble_config
                obs_vectors: Dict[str, ObsVector] = {}
                for obstype, obs_name, values in obs_config_content:
                    if obstype == ObservationType.HISTORY:
                        obs_vectors.update(
                            **cls._handle_history_observation(
                                ensemble_config,
                                values,  # type: ignore
                                obs_name,
                                std_cutoff,
                                history,
                                time_len,
                            )
                        )
                    elif obstype == ObservationType.SUMMARY:
                        obs_vectors.update(
                            **cls._handle_summary_observation(
                                ensemble_config,
                                values,  # type: ignore
                                obs_name,
                                obs_time_list,
                            )
                        )
                    elif obstype == ObservationType.GENERAL:
                        obs_vectors.update(
                            **cls._handle_general_observation(
                                ensemble_config,
                                values,  # type: ignore
                                obs_name,
                                obs_time_list,
                            )
                        )
                    else:
                        raise ValueError(f"Unknown ObservationType {obstype}")

                for state_kw in set(o.data_key for o in obs_vectors.values()):
                    assert state_kw in ensemble_config.response_configs
                return EnkfObs(obs_vectors, obs_time_list)
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
        return EnkfObs({}, obs_time_list)

    def get_dataset(self, key: str) -> Tuple[str, xr.Dataset]:
        return self[key].to_dataset(self, [])
