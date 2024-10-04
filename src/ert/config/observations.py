import os
from dataclasses import dataclass
from datetime import datetime, timedelta
from typing import TYPE_CHECKING, Dict, Iterator, List, Optional, Tuple, Union

import numpy as np
import xarray as xr

from ert.validation import rangestring_to_list

from .enkf_observation_implementation_type import EnkfObservationImplementationType
from .gen_data_config import GenDataConfig
from .general_observation import GenObservation
from .observation_vector import ObsVector
from .parsing import ConfigWarning, HistorySource
from .parsing.observations_parser import (
    DateValues,
    ErrorValues,
    GenObsValues,
    HistoryValues,
    ObservationConfigError,
    SummaryValues,
)
from .summary_observation import SummaryObservation

if TYPE_CHECKING:
    import numpy.typing as npt

    from .ensemble_config import EnsembleConfig

DEFAULT_TIME_DELTA = timedelta(seconds=30)


def history_key(key: str) -> str:
    keyword, *rest = key.split(":")
    return ":".join([keyword + "H", *rest])


@dataclass
class EnkfObs:
    obs_vectors: Dict[str, ObsVector]
    obs_time: List[datetime]

    def __post_init__(self) -> None:
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
        error_dict: ErrorValues,
    ) -> "npt.NDArray[np.double]":
        values = np.asarray(values)
        error_mode = error_dict.error_mode
        error_min = error_dict.error_min
        error = error_dict.error
        if error_mode == "ABS":
            return np.full(values.shape, error)
        elif error_mode == "REL":
            return np.abs(values) * error
        elif error_mode == "RELMIN":
            return np.maximum(np.abs(values) * error, np.full(values.shape, error_min))
        raise ObservationConfigError(f"Unknown error mode {error_mode}", error_mode)

    @classmethod
    def _handle_history_observation(
        cls,
        ensemble_config: "EnsembleConfig",
        history_observation: HistoryValues,
        summary_key: str,
        history_type: HistorySource,
        time_len: int,
    ) -> Dict[str, ObsVector]:
        refcase = ensemble_config.refcase
        if refcase is None:
            raise ObservationConfigError("REFCASE is required for HISTORY_OBSERVATION")
        error = history_observation.error

        if history_type == HistorySource.REFCASE_HISTORY:
            local_key = history_key(summary_key)
        else:
            local_key = summary_key
        if local_key is None:
            return {}
        if local_key not in refcase.keys:
            return {}
        values = refcase.values[refcase.keys.index(local_key)]
        std_dev = cls._handle_error_mode(values, history_observation)
        for segment_name, segment_instance in history_observation.segment:
            start = segment_instance.start
            stop = segment_instance.stop
            if start < 0:
                ConfigWarning.warn(
                    f"Segment {segment_name} out of bounds."
                    " Truncating start of segment to 0.",
                    segment_name,
                )
                start = 0
            if stop >= time_len:
                ConfigWarning.warn(
                    f"Segment {segment_name} out of bounds. Truncating"
                    f" end of segment to {time_len - 1}.",
                    segment_name,
                )
                stop = time_len - 1
            if start > stop:
                ConfigWarning.warn(
                    f"Segment {segment_name} start after stop. Truncating"
                    f" end of segment to {start}.",
                    segment_name,
                )
                stop = start
            if np.size(std_dev[start:stop]) == 0:
                ConfigWarning.warn(
                    f"Segment {segment_name} does not"
                    " contain any time steps. The interval "
                    f"[{start}, {stop}) does not intersect with steps in the"
                    "time map.",
                    segment_name,
                )
            std_dev[start:stop] = cls._handle_error_mode(
                values[start:stop],
                segment_instance,
            )
        data: Dict[Union[int, datetime], Union[GenObservation, SummaryObservation]] = {}
        for date, error, value in zip(refcase.dates, std_dev, values):
            data[date] = SummaryObservation(summary_key, summary_key, value, error)

        return {
            summary_key: ObsVector(
                EnkfObservationImplementationType.SUMMARY_OBS,
                summary_key,
                "summary",
                data,
            )
        }

    @staticmethod
    def _get_time(date_dict: DateValues, start_time: datetime) -> Tuple[datetime, str]:
        if date_dict.date is not None:
            date_str = date_dict.date
            try:
                return datetime.fromisoformat(date_str), f"DATE={date_str}"
            except ValueError:
                try:
                    date = datetime.strptime(date_str, "%d/%m/%Y")
                    ConfigWarning.warn(
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

        if date_dict.days is not None:
            days = date_dict.days
            return start_time + timedelta(days=days), f"DAYS={days}"
        if date_dict.hours is not None:
            hours = date_dict.hours
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
        date_dict: DateValues,
        obs_name: str,
        time_map: List[datetime],
        has_refcase: bool,
    ) -> int:
        if date_dict.restart is not None:
            return date_dict.restart
        if not time_map:
            raise ObservationConfigError.with_context(
                f"Missing REFCASE or TIME_MAP for observations: {obs_name}",
                obs_name,
            )

        try:
            time, date_str = EnkfObs._get_time(date_dict, time_map[0])
        except ObservationConfigError:
            raise
        except ValueError as err:
            raise ObservationConfigError.with_context(
                f"Failed to parse date of {obs_name}", obs_name
            ) from err

        try:
            return EnkfObs._find_nearest(time_map, time)
        except IndexError as err:
            raise ObservationConfigError.with_context(
                f"Could not find {time} ({date_str}) in "
                f"the time map for observations {obs_name}"
                + (
                    "The time map is set from the REFCASE keyword. Either "
                    "the REFCASE has an incorrect/missing date, or the observation "
                    "is given an incorrect date.)"
                    if has_refcase
                    else " (The time map is set from the TIME_MAP "
                    "keyword. Either the time map file has an "
                    "incorrect/missing date, or the  observation is given an "
                    "incorrect date."
                ),
                obs_name,
            ) from err

    @staticmethod
    def _make_value_and_std_dev(
        observation_dict: SummaryValues,
    ) -> Tuple[float, float]:
        value = observation_dict.value
        return (
            value,
            float(
                EnkfObs._handle_error_mode(
                    np.array(value),
                    observation_dict,
                )
            ),
        )

    @classmethod
    def _handle_summary_observation(
        cls,
        summary_dict: SummaryValues,
        obs_key: str,
        time_map: List[datetime],
        has_refcase: bool,
    ) -> Dict[str, ObsVector]:
        summary_key = summary_dict.key
        value, std_dev = cls._make_value_and_std_dev(summary_dict)

        try:
            if summary_dict.date is not None and not time_map:
                # We special case when the user has provided date in SUMMARY_OBS
                # and not REFCASE or time_map so that we dont change current behavior.
                try:
                    date = datetime.fromisoformat(summary_dict.date)
                except ValueError as err:
                    raise ValueError("Please use ISO date format YYYY-MM-DD.") from err
                restart = None
            else:
                restart = cls._get_restart(summary_dict, obs_key, time_map, has_refcase)
                date = time_map[restart]
        except ValueError as err:
            raise ObservationConfigError.with_context(
                f"Problem with date in summary observation {obs_key}: " + str(err),
                obs_key,
            ) from err

        if restart == 0:
            raise ObservationConfigError.with_context(
                "It is unfortunately not possible to use summary "
                "observations from the start of the simulation. "
                f"Problem with observation {obs_key}"
                f"{' at ' + str(cls._get_time(summary_dict, time_map[0])) if summary_dict.restart is None else ''}",
                obs_key,
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
                raise ObservationConfigError.with_context(
                    "Expected even number of values in GENERAL_OBSERVATION", obs_file
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
                indices = np.array(
                    sorted(rangestring_to_list(data_index)), dtype=np.int32
                )
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
        has_refcase: bool,
    ) -> Dict[str, ObsVector]:
        response_key = general_observation.data
        if not ensemble_config.hasNodeGenData(response_key):
            ConfigWarning.warn(
                f"No GEN_DATA with name: {response_key} found - ignoring observation {obs_key}",
                response_key,
            )
            return {}

        if all(
            getattr(general_observation, key) is None
            for key in ["restart", "date", "days", "hours"]
        ):
            # The user has not provided RESTART or DATE, this is legal
            # for GEN_DATA, so we default it to None
            restart = None
        else:
            restart = cls._get_restart(
                general_observation, obs_key, time_map, has_refcase
            )

        gen_data_config = ensemble_config.response_configs.get("gen_data", None)
        assert isinstance(gen_data_config, GenDataConfig)
        if response_key not in gen_data_config.keys:
            ConfigWarning.warn(
                f"Observation {obs_key} on GEN_DATA key {response_key}, but GEN_DATA"
                f" key {response_key} is non-existing"
            )
            return {}

        _, report_steps = gen_data_config.get_args_for_key(response_key)

        response_report_steps = [] if report_steps is None else report_steps
        if (restart is None and response_report_steps) or (
            restart is not None and restart not in response_report_steps
        ):
            ConfigWarning.warn(
                f"The GEN_DATA node:{response_key} is not configured to load from"
                f" report step:{restart} for the observation:{obs_key}"
                " - The observation will be ignored",
                response_key,
            )
            return {}

        restart = 0 if restart is None else restart
        index_list = general_observation.index_list
        index_file = general_observation.index_file
        if index_list is not None and index_file is not None:
            raise ObservationConfigError.with_context(
                f"GENERAL_OBSERVATION {obs_key} has both INDEX_FILE and INDEX_LIST.",
                obs_key,
            )
        indices = index_list if index_list is not None else index_file
        try:
            return {
                obs_key: ObsVector(
                    EnkfObservationImplementationType.GEN_OBS,
                    obs_key,
                    response_key,
                    {
                        restart: cls._create_gen_obs(
                            (
                                (
                                    general_observation.value,
                                    general_observation.error,
                                )
                                if general_observation.value is not None
                                and general_observation.error is not None
                                else None
                            ),
                            general_observation.obs_file,
                            indices,
                        ),
                    },
                )
            }
        except ValueError as err:
            raise ObservationConfigError.with_context(str(err), obs_key) from err

    def __repr__(self) -> str:
        return f"EnkfObs({self.obs_vectors}, {self.obs_time})"
