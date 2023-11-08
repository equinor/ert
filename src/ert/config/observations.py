import os
from datetime import datetime, timedelta
from typing import TYPE_CHECKING, Dict, Iterator, List, Literal, Optional, Tuple, Union

import numpy as np
import xarray as xr
from resdata.summary import SummaryVarType
from resdata.util.util import IntVector

from ert._clib.enkf_obs import read_from_refcase

from .enkf_observation_implementation_type import EnkfObservationImplementationType
from .gen_data_config import GenDataConfig
from .general_observation import GenObservation
from .observation_vector import ObsVector
from .parsing import ConfigWarning, HistorySource
from .parsing.observations_parser import (
    DateDict,
    GenObsValues,
    HistoryValues,
    ObservationConfigError,
    SummaryValues,
)
from .summary_config import SummaryConfig
from .summary_observation import SummaryObservation

if TYPE_CHECKING:
    import numpy.typing as npt

    from .ensemble_config import EnsembleConfig

DEFAULT_TIME_DELTA = timedelta(seconds=30)


class EnkfObs:
    def __init__(self, obs_vectors: Dict[str, ObsVector], obs_time: List[datetime]):
        self.obs_vectors = obs_vectors
        self.obs_time = obs_time
        self.datasets: Dict[str, xr.Dataset] = {
            name: obs.to_dataset([]) for name, obs in sorted(self.obs_vectors.items())
        }

    def __len__(self) -> int:
        return len(self.obs_vectors)

    def __contains__(self, key: str) -> bool:
        return key in self.obs_vectors

    def __iter__(self) -> Iterator[ObsVector]:
        return iter(self.obs_vectors.values())

    def __getitem__(self, key: str) -> ObsVector:
        return self.obs_vectors[key]

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, EnkfObs):
            return False
        # Datasets contains the full observations, so if they are equal, everything is
        return self.datasets == other.datasets

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
            return np.abs(values) * error
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

        refcase = ensemble_config.refcase
        if refcase is None:
            raise ObservationConfigError("REFCASE is required for HISTORY_OBSERVATION")
        if history_type is None:
            raise ValueError("Need a history type in order to use history observations")

        if summary_key not in response_config.keys:
            response_config.keys.append(summary_key)
        error = history_observation["ERROR"]
        error_min = history_observation["ERROR_MIN"]
        error_mode = history_observation["ERROR_MODE"]

        if history_type == HistorySource.REFCASE_HISTORY:
            var_type = refcase.var_type(summary_key)
            local_key = None
            if var_type in [
                SummaryVarType.RD_SMSPEC_WELL_VAR,
                SummaryVarType.RD_SMSPEC_GROUP_VAR,
            ]:
                summary_node = refcase.smspec_node(summary_key)
                local_key = summary_node.keyword + "H:" + summary_node.wgname
            elif var_type == SummaryVarType.RD_SMSPEC_FIELD_VAR:
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
                ConfigWarning.ert_context_warn(
                    f"Segment {segment_name} out of bounds."
                    " Truncating start of segment to 0.",
                    segment_name,
                )
                start = 0
            if stop >= time_len:
                ConfigWarning.ert_context_warn(
                    f"Segment {segment_name} out of bounds. Truncating"
                    f" end of segment to {time_len - 1}.",
                    segment_name,
                )
                stop = time_len - 1
            if start > stop:
                ConfigWarning.ert_context_warn(
                    f"Segment {segment_name} start after stop. Truncating"
                    f" end of segment to {start}.",
                    segment_name,
                )
                stop = start
            if np.size(std_dev[start:stop]) == 0:
                ConfigWarning.ert_context_warn(
                    f"Segment {segment_name} does not"
                    " contain any time steps. The interval "
                    f"[{start}, {stop}) does not intersect with steps in the"
                    "time map.",
                    segment_name,
                )
            std_dev[start:stop] = cls._handle_error_mode(
                values[start:stop],
                segment_instance["ERROR"],
                segment_instance["ERROR_MIN"],
                segment_instance["ERROR_MODE"],
            )
        data: Dict[Union[int, datetime], Union[GenObservation, SummaryObservation]] = {}
        dates = [
            datetime(date.year, date.month, date.day) for date in refcase.report_dates
        ]
        for i, (good, error, value) in enumerate(zip(valid, std_dev, values)):
            if good:
                if error <= std_cutoff:
                    ConfigWarning.ert_context_warn(
                        "Too small observation error in observation"
                        f" {summary_key}:{i} - ignored",
                        summary_key,
                    )
                    continue
                data[dates[i - 1]] = SummaryObservation(
                    summary_key, summary_key, value, error
                )

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
                    ConfigWarning.ert_context_warn(
                        f"Deprecated time format {date_str}."
                        " Please use ISO date format YYYY-MM-DD",
                        date_str,
                    )
                    return date, f"DATE={date_str}"
                except ValueError as err:
                    raise ObservationConfigError.with_context(
                        f"Unsupported date format {date_str}."
                        " Please use ISO date format",
                        date_str,
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
            if diff < threshold and (nearest_diff is None or nearest_diff > diff):
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
            if not time_map:
                raise ObservationConfigError.with_context(
                    f"Missing REFCASE or TIME_MAP for observations: {obs_name}",
                    obs_name,
                )
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
        summary_config = ensemble_config["summary"]
        assert isinstance(summary_config, SummaryConfig)
        if summary_key not in summary_config.keys:
            summary_config.keys.append(summary_key)
        value, std_dev = cls._make_value_and_std_dev(summary_dict)
        try:

            def str_to_datetime(date_str: str) -> datetime:
                try:
                    return datetime.fromisoformat(date_str)
                except ValueError as err:
                    raise ValueError("Please use ISO date format YYYY-MM-DD.") from err

            if "DATE" in summary_dict and not time_map:
                # We special case when the user has provided date in SUMMARY_OBS
                # and not REFCASE so that we dont change current behavior.
                date = str_to_datetime(summary_dict["DATE"])
                restart = None
            else:
                restart = cls._get_restart(summary_dict, obs_key, time_map)
                date = time_map[restart]
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
                {date: SummaryObservation(summary_key, obs_key, value, std_dev)},
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
            try:
                file_values = np.loadtxt(obs_file, delimiter=None).ravel()
            except ValueError as err:
                raise ObservationConfigError.with_context(
                    f"Failed to read OBS_FILE {obs_file}: {err}", obs_file
                ) from err
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
                indices = np.loadtxt(data_index, delimiter=None, dtype=int).ravel()
            else:
                indices = np.array(IntVector.active_list(data_index), dtype=np.int32)
        else:
            indices = np.arange(len(values))
        std_scaling = np.full(len(values), 1.0)
        if len({len(stds), len(values), len(indices)}) != 1:
            raise ObservationConfigError.with_context(
                f"Values ({values}), error ({stds}) and "
                f"index list ({indices}) must be of equal length",
                obs_file if obs_file is not None else "",
            )
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
            ConfigWarning.ert_context_warn(
                f"Ensemble key {state_kw} does not exist"
                f" - ignoring observation {obs_key}",
                state_kw,
            )
            return {}
        config_node = ensemble_config.getNode(state_kw)
        try:
            if not any(
                key in general_observation
                for key in ["RESTART", "DATE", "DAYS", "HOURS"]
            ):
                # The user has not provided RESTART or DATE, this is legal
                # for GEN_DATA, so we default it to None
                restart = None
            else:
                restart = cls._get_restart(general_observation, obs_key, time_map)
        except ValueError as err:
            raise ObservationConfigError.with_context(
                f"Problem with date in general observation {obs_key}: " + str(err),
                obs_key,
            ) from err
        if not isinstance(config_node, GenDataConfig):
            ConfigWarning.ert_context_warn(
                f"{state_kw} has implementation type:"
                f"'{type(config_node)}' - "
                f"expected:'GEN_DATA' in observation:{obs_key}."
                "The observation will be ignored",
                obs_key,
            )
            return {}
        response_report_steps = (
            [] if config_node.report_steps is None else config_node.report_steps
        )
        if (restart is None and response_report_steps) or (
            restart is not None and restart not in response_report_steps
        ):
            ConfigWarning.ert_context_warn(
                f"The GEN_DATA node:{state_kw} is not configured to load from"
                f" report step:{restart} for the observation:{obs_key}"
                " - The observation will be ignored",
                state_kw,
            )
            return {}
        restart = 0 if restart is None else restart
        index_list = general_observation.get("INDEX_FILE")
        index_file = general_observation.get("INDEX_LIST")
        if index_list is not None and index_file is not None:
            raise ObservationConfigError.with_context(
                f"GENERAL_OBSERVATION {obs_key} has both INDEX_FILE and INDEX_LIST.",
                obs_key,
            )
        indices = index_list if index_list is not None else index_file
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
                        general_observation.get("OBS_FILE"),
                        indices,
                    ),
                },
            )
        }

    def __repr__(self) -> str:
        return f"EnkfObs({self.obs_vectors}, {self.obs_time})"
