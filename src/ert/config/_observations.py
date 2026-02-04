from __future__ import annotations

import logging
import os
from collections.abc import Sequence
from datetime import datetime
from enum import StrEnum
from typing import Annotated, Any, Literal, Self, assert_never

import numpy as np
import pandas as pd
from pydantic import BaseModel, Field

from ..validation import rangestring_to_list
from .parsing import (
    ConfigWarning,
    ErrorInfo,
    ObservationConfigError,
    ObservationDict,
    ObservationType,
)

logger = logging.getLogger(__name__)


class ErrorModes(StrEnum):
    REL = "REL"
    ABS = "ABS"
    RELMIN = "RELMIN"


class _SummaryValues(BaseModel):
    type: Literal["summary_observation"] = "summary_observation"
    name: str
    value: float
    error: float
    key: str  #: The :term:`summary key` in the summary response
    date: str
    east: float | None = None
    north: float | None = None
    radius: float | None = None


def _parse_date(date_str: str) -> datetime:
    try:
        return datetime.fromisoformat(date_str)
    except ValueError:
        try:
            date = datetime.strptime(date_str, "%d/%m/%Y")
        except ValueError as err:
            raise ObservationConfigError.with_context(
                f"Unsupported date format {date_str}. Please use ISO date format",
                date_str,
            ) from err
        else:
            ConfigWarning.warn(
                f"Deprecated time format {date_str}."
                " Please use ISO date format YYYY-MM-DD",
                date_str,
            )
            return date


class SummaryObservation(_SummaryValues):
    @classmethod
    def from_obs_dict(
        cls, directory: str, observation_dict: ObservationDict
    ) -> list[Self]:
        error_mode = ErrorModes.ABS
        summary_key = None

        date: str | None = None
        float_values: dict[str, float] = {"ERROR_MIN": 0.1}
        localization_dict: dict[LOCALIZATION_KEYS, float | None] = {}
        for key, value in observation_dict.items():
            match key:
                case "type" | "name":
                    pass
                case "ERROR" | "ERROR_MIN":
                    float_values[str(key)] = validate_positive_float(
                        value, key, strictly_positive=True
                    )
                case "DAYS" | "HOURS" | "RESTART":
                    raise ObservationConfigError.with_context(
                        (
                            "SUMMARY_OBSERVATION must use DATE to specify "
                            "date, DAYS | HOURS is no longer allowed. "
                            "Please run:\n ert convert_observations "
                            "<your_ert_config.ert>\nto migrate the observation config "
                            "to use the correct format."
                        ),
                        key,
                    )
                case "VALUE":
                    float_values[str(key)] = validate_float(value, key)
                case "ERROR_MODE":
                    error_mode = validate_error_mode(value)
                case "KEY":
                    summary_key = value
                case "DATE":
                    date = value
                case "LOCALIZATION":
                    validate_localization(value, observation_dict["name"])
                    extract_localization_values(localization_dict, value, key)
                case _:
                    raise _unknown_key_error(str(key), observation_dict["name"])
        if "VALUE" not in float_values:
            raise _missing_value_error(observation_dict["name"], "VALUE")
        if summary_key is None:
            raise _missing_value_error(observation_dict["name"], "KEY")
        if "ERROR" not in float_values:
            raise _missing_value_error(observation_dict["name"], "ERROR")

        assert date is not None
        # Raise errors if the date is off
        parsed_date: datetime = _parse_date(date)
        standardized_date = parsed_date.date().isoformat()

        value = float_values["VALUE"]
        input_error = float_values["ERROR"]
        error_min = float_values["ERROR_MIN"]

        error = input_error
        match error_mode:
            case ErrorModes.ABS:
                error = validate_positive_float(
                    np.abs(input_error), summary_key, strictly_positive=True
                )
            case ErrorModes.REL:
                error = validate_positive_float(
                    np.abs(value) * input_error, summary_key, strictly_positive=True
                )
            case ErrorModes.RELMIN:
                error = validate_positive_float(
                    np.maximum(np.abs(value) * input_error, error_min),
                    summary_key,
                    strictly_positive=True,
                )
            case default:
                assert_never(default)

        obs_instance = cls(
            name=observation_dict["name"],
            error=error,
            key=summary_key,
            value=value,
            east=localization_dict.get("east"),
            north=localization_dict.get("north"),
            radius=localization_dict.get("radius"),
            date=standardized_date,
        )
        # Bypass pydantic discarding context
        # only relevant for ERT config surfacing validation errors
        # irrelevant for runmodels etc.
        obs_instance.name = observation_dict["name"]

        return [obs_instance]


class _GeneralObservation(BaseModel):
    type: Literal["general_observation"] = "general_observation"
    name: str
    data: str
    value: float
    error: float
    restart: int
    index: int
    east: float | None = None
    north: float | None = None
    radius: float | None = None


class GeneralObservation(_GeneralObservation):
    @classmethod
    def from_obs_dict(
        cls, directory: str, observation_dict: ObservationDict
    ) -> list[Self]:
        try:
            data = observation_dict["DATA"]
        except KeyError as err:
            raise _missing_value_error(observation_dict["name"], "DATA") from err

        allowed = {
            "type",
            "name",
            "RESTART",
            "VALUE",
            "ERROR",
            "DATE",
            "DAYS",
            "HOURS",
            "INDEX_LIST",
            "OBS_FILE",
            "INDEX_FILE",
            "DATA",
        }

        extra = set(observation_dict.keys()) - allowed
        if extra:
            raise _unknown_key_error(str(next(iter(extra))), observation_dict["name"])

        if any(k in observation_dict for k in ("DATE", "DAYS", "HOURS")):
            bad_key = next(
                k for k in ("DATE", "DAYS", "HOURS") if k in observation_dict
            )
            raise ObservationConfigError.with_context(
                (
                    "GENERAL_OBSERVATION must use RESTART to specify "
                    "report step. Please run:\n ert convert_observations "
                    "<your_ert_config.ert>\nto migrate the observation config "
                    "to use the correct format."
                ),
                bad_key,
            )

        if "OBS_FILE" in observation_dict and (
            "VALUE" in observation_dict or "ERROR" in observation_dict
        ):
            raise ObservationConfigError.with_context(
                "GENERAL_OBSERVATION cannot contain both VALUE/ERROR and OBS_FILE",
                observation_dict["name"],
            )

        if "INDEX_FILE" in observation_dict and "INDEX_LIST" in observation_dict:
            raise ObservationConfigError.with_context(
                (
                    "GENERAL_OBSERVATION "
                    f"{observation_dict['name']} has both INDEX_FILE and INDEX_LIST."
                ),
                observation_dict["name"],
            )

        restart = (
            validate_positive_int(observation_dict["RESTART"], "RESTART")
            if "RESTART" in observation_dict
            else 0
        )

        if "OBS_FILE" not in observation_dict:
            if "VALUE" in observation_dict and "ERROR" not in observation_dict:
                raise ObservationConfigError.with_context(
                    f"For GENERAL_OBSERVATION {observation_dict['name']}, with"
                    f" VALUE = {observation_dict['VALUE']}, ERROR must also be given.",
                    observation_dict["name"],
                )

            if "VALUE" not in observation_dict and "ERROR" not in observation_dict:
                raise ObservationConfigError.with_context(
                    "GENERAL_OBSERVATION must contain either VALUE "
                    "and ERROR or OBS_FILE",
                    context=observation_dict["name"],
                )

            obs_instance = cls(
                name=observation_dict["name"],
                data=data,
                value=validate_float(observation_dict["VALUE"], "VALUE"),
                error=validate_positive_float(
                    observation_dict["ERROR"], "ERROR", strictly_positive=True
                ),
                restart=restart,
                index=0,
            )
            # Bypass pydantic discarding context
            # only relevant for ERT config surfacing validation errors
            # irrelevant for runmodels etc.
            obs_instance.name = observation_dict["name"]
            return [obs_instance]

        obs_filename = _resolve_path(directory, observation_dict["OBS_FILE"])
        try:
            file_values = np.loadtxt(obs_filename, delimiter=None).ravel()
        except ValueError as err:
            raise ObservationConfigError.with_context(
                f"Failed to read OBS_FILE {obs_filename}: {err}", obs_filename
            ) from err
        if len(file_values) % 2 != 0:
            raise ObservationConfigError.with_context(
                "Expected even number of values in GENERAL_OBSERVATION",
                obs_filename,
            )
        values = file_values[::2]
        stds = file_values[1::2]

        if "INDEX_FILE" in observation_dict:
            idx_file = _resolve_path(directory, observation_dict["INDEX_FILE"])
            indices = np.loadtxt(idx_file, delimiter=None, dtype=np.int32).ravel()
        elif "INDEX_LIST" in observation_dict:
            indices = np.array(
                sorted(rangestring_to_list(observation_dict["INDEX_LIST"])),
                dtype=np.int32,
            )
        else:
            indices = np.arange(len(values), dtype=np.int32)

        if len({len(stds), len(values), len(indices)}) != 1:
            raise ObservationConfigError.with_context(
                (
                    "Values ("
                    f"{values}), error ({stds}) and index list ({indices}) "
                    "must be of equal length"
                ),
                observation_dict["name"],
            )

        if np.any(stds <= 0):
            raise ObservationConfigError.with_context(
                "Observation uncertainty must be strictly > 0",
                observation_dict["name"],
            )

        obs_instances: list[Self] = []
        for _pos, (val, std, idx) in enumerate(zip(values, stds, indices, strict=True)):
            # index should reflect the index provided by INDEX_FILE / INDEX_LIST
            inst = cls(
                name=observation_dict["name"],
                data=data,
                value=float(val),
                error=float(std),
                restart=restart,
                index=int(idx),
            )
            # Bypass pydantic discarding context
            # only relevant for ERT config surfacing validation errors
            # irrelevant for runmodels etc.
            inst.name = observation_dict["name"]
            obs_instances.append(inst)
        return obs_instances


class RFTObservation(BaseModel):
    """Represents an RFT (Repeat Formation Tester) observation.

    RFT observations are used to condition on pressure, saturation, or other
    properties measured at specific well locations and times.
    """

    type: Literal["rft_observation"] = "rft_observation"
    name: str
    well: str
    date: str
    property: str
    value: float
    error: float
    north: float
    east: float
    tvd: float
    zone: str | None = None

    @classmethod
    def from_csv(
        cls,
        directory: str,
        observation_dict: ObservationDict,
        filename: str,
        observed_property: str = "PRESSURE",
    ) -> list[Self]:
        """Create RFT observations from a CSV file.

        The CSV file must contain the following columns: WELL_NAME, DATE,
        ERROR, NORTH, EAST, TVD, and a column for the observed property
        (e.g., PRESSURE, SWAT). An optional ZONE column may also be present.

        Args:
            directory: Base directory for resolving relative file paths.
            observation_dict: Dictionary containing the observation configuration.
            filename: Path to the CSV file containing RFT observations.
            observed_property: Property to observe (default: PRESSURE).

        Returns:
            List of RFTObservation instances created from the CSV file.

        Raises:
            ObservationConfigError: If the file is missing, inaccessible,
                lacks required columns, or contains invalid observation values
                (value=-1 and error=0).
        """
        if not os.path.isabs(filename):
            filename = os.path.join(directory, filename)
        if not os.path.exists(filename):
            raise ObservationConfigError.with_context(
                f"The CSV file ({filename}) does not exist or is not accessible.",
                filename,
            )
        csv_file = pd.read_csv(
            filename,
            encoding="utf-8",
            on_bad_lines="error",
        )

        required_columns = {
            "WELL_NAME",
            "DATE",
            observed_property,
            "ERROR",
            "NORTH",
            "EAST",
            "TVD",
        }
        missing_required_columns = required_columns - set(csv_file.keys())
        if missing_required_columns:
            raise ObservationConfigError.with_context(
                f"The rft observations file {filename} is missing required column(s) "
                f"{', '.join(sorted(missing_required_columns))}.",
                filename,
            )

        rft_observations = []
        invalid_observations = []
        for row in csv_file.itertuples(index=True):
            rft_observation = cls(
                name=f"{observation_dict['name']}[{row.Index}]",
                well=str(row.WELL_NAME),
                date=str(row.DATE),
                property=observed_property,
                value=validate_float(
                    str(getattr(row, observed_property)), observed_property
                ),
                error=validate_float(str(row.ERROR), "ERROR"),
                north=validate_float(str(row.NORTH), "NORTH"),
                east=validate_float(str(row.EAST), "EAST"),
                tvd=validate_float(str(row.TVD), "TVD"),
                zone=row.ZONE if "ZONE" in csv_file else None,
            )
            # A value of -1 and error of 0 is used by fmu.tools.rms create_rft_ertobs to
            # indicate missing data. If encountered in an rft observations csv file
            # it should raise an error and ask the user to remove invalid observations.
            if rft_observation.value == -1 and rft_observation.error == 0:
                invalid_observations.append(rft_observation)
            else:
                rft_observations.append(rft_observation)

        if invalid_observations:
            well_list = "\n - ".join(
                [
                    f"{observation.well} at date {observation.date}"
                    for observation in invalid_observations
                ]
            )
            raise ObservationConfigError.with_context(
                (
                    f"Invalid value=-1 and error=0 detected in {filename} for "
                    f"well(s):\n - {well_list}\n"
                    "The invalid observation(s) must be removed from the file."
                ),
                filename,
            )
        return rft_observations

    @classmethod
    def from_obs_dict(
        cls, directory: str, observation_dict: ObservationDict
    ) -> list[Self]:
        """Create RFT observations from an observation dictionary.

        Supports two modes:
        1. CSV mode: Load observations from a CSV file specified by the CSV key.
        2. Direct mode: Create a single observation from individual keys
           (WELL, PROPERTY, VALUE, ERROR, DATE, NORTH, EAST, TVD, ZONE).

        Args:
            directory: Base directory for resolving relative file paths.
            observation_dict: Dictionary containing the observation configuration.

        Returns:
            List of RFTObservation instances. Returns multiple observations when
            loading from CSV, or a single observation when using direct mode.

        Raises:
            ObservationConfigError: If required keys are missing or invalid.
        """
        csv_filename = None
        well = None
        observed_property = None
        observed_value = None
        error = None
        date = None
        north = None
        east = None
        tvd = None
        zone = None
        for key, value in observation_dict.items():
            match key:
                case "type" | "name":
                    pass
                case "WELL":
                    well = value
                case "PROPERTY":
                    observed_property = value
                case "VALUE":
                    observed_value = validate_float(value, key)
                case "ERROR":
                    error = validate_float(value, key)
                case "DATE":
                    date = value
                case "NORTH":
                    north = validate_float(value, key)
                case "EAST":
                    east = validate_float(value, key)
                case "TVD":
                    tvd = validate_float(value, key)
                case "CSV":
                    csv_filename = value
                case "ZONE":
                    zone = value
                case _:
                    raise _unknown_key_error(str(key), observation_dict["name"])
        if csv_filename is not None:
            return cls.from_csv(
                directory,
                observation_dict,
                csv_filename,
                observed_property or "PRESSURE",
            )
        if well is None:
            raise _missing_value_error(observation_dict["name"], "WELL")
        if observed_value is None:
            raise _missing_value_error(observation_dict["name"], "VALUE")
        if observed_property is None:
            raise _missing_value_error(observation_dict["name"], "PROPERTY")
        if error is None:
            raise _missing_value_error(observation_dict["name"], "ERROR")
        if date is None:
            raise _missing_value_error(observation_dict["name"], "DATE")
        if north is None:
            raise _missing_value_error(observation_dict["name"], "NORTH")
        if east is None:
            raise _missing_value_error(observation_dict["name"], "EAST")
        if tvd is None:
            raise _missing_value_error(observation_dict["name"], "TVD")

        obs_instance = cls(
            name=observation_dict["name"],
            well=well,
            property=observed_property,
            value=observed_value,
            error=error,
            date=date,
            north=north,
            east=east,
            tvd=tvd,
            zone=zone,
        )

        # Bypass pydantic discarding context
        # only relevant for ERT config surfacing validation errors
        # irrelevant for runmodels etc.
        obs_instance.name = observation_dict["name"]

        return [obs_instance]


class BreakthroughObservation(BaseModel):
    type: Literal["breakthrough"] = "breakthrough"
    name: str
    key: str
    date: datetime
    error: float
    threshold: float
    north: float | None
    east: float | None
    radius: float | None

    @classmethod
    def from_obs_dict(cls, directory: str, obs_dict: ObservationDict) -> list[Self]:
        summary_key = None
        date = None
        error = None
        threshold = None
        localization_dict: dict[LOCALIZATION_KEYS, float | None] = {}
        for key, value in obs_dict.items():
            match key:
                case "type" | "name":
                    pass
                case "KEY":
                    summary_key = value
                case "DATE":
                    date = value
                case "ERROR":
                    error = validate_float(value, key)
                case "THRESHOLD":
                    threshold = validate_float(value, key)
                case "LOCALIZATION":
                    validate_localization(value, obs_dict["name"])
                    extract_localization_values(localization_dict, value, key)
                case _:
                    raise _unknown_key_error(str(key), value)

        if summary_key is None:
            raise _missing_value_error(obs_dict["name"], "KEY")
        if date is None:
            raise _missing_value_error(obs_dict["name"], "DATE")
        if error is None:
            raise _missing_value_error(obs_dict["name"], "ERROR")
        if threshold is None:
            raise _missing_value_error(obs_dict["name"], "THRESHOLD")

        return [
            cls(
                name=obs_dict["name"],
                key=summary_key,
                date=date,
                error=error,
                threshold=threshold,
                north=localization_dict.get("north"),
                east=localization_dict.get("east"),
                radius=localization_dict.get("radius"),
            )
        ]


Observation = Annotated[
    (
        SummaryObservation
        | GeneralObservation
        | RFTObservation
        | BreakthroughObservation
    ),
    Field(discriminator="type"),
]

_TYPE_TO_CLASS: dict[ObservationType, type[Observation]] = {
    ObservationType.SUMMARY: SummaryObservation,
    ObservationType.GENERAL: GeneralObservation,
    ObservationType.RFT: RFTObservation,
    ObservationType.BREAKTHROUGH: BreakthroughObservation,
}

LOCALIZATION_KEYS = Literal["east", "north", "radius"]


def make_observations(
    directory: str, observation_dicts: Sequence[ObservationDict]
) -> list[Observation]:
    """Takes observation dicts and returns validated observations.

    Param:
        directory: The name of the directory the observation config is located in.
            Used to disambiguate relative paths.
        inp: The collection of statements to validate.
    """
    result: list[Observation] = []
    error_list: list[ErrorInfo | ObservationConfigError] = []
    for obs_dict in observation_dicts:
        if obs_dict["type"] == ObservationType.HISTORY:
            msg = (
                "HISTORY_OBSERVATION is deprecated, and must be specified "
                "as SUMMARY_OBSERVATION. Run"
                " ert convert_observations <ert_config.ert> to convert your "
                "observations automatically"
            )
            logger.error(msg)
            error_list.append(ObservationConfigError(msg))
            continue
        try:
            result.extend(
                _TYPE_TO_CLASS[obs_dict["type"]].from_obs_dict(directory, obs_dict)
            )
        except KeyError as err:
            raise _unknown_observation_type_error(obs_dict) from err
        except ObservationConfigError as err:
            error_list.extend(err.errors)

    if error_list:
        raise ObservationConfigError.from_collected(error_list)

    return result


def validate_error_mode(inp: str) -> ErrorModes:
    if inp == "REL":
        return ErrorModes.REL
    if inp == "ABS":
        return ErrorModes.ABS
    if inp == "RELMIN":
        return ErrorModes.RELMIN
    raise ObservationConfigError.with_context(
        f'Unexpected ERROR_MODE {inp}. Failed to validate "{inp}"', inp
    )


def validate_float(val: str, key: str) -> float:
    try:
        return float(val)
    except ValueError as err:
        raise _conversion_error(key, val, "float") from err


def validate_int(val: str, key: str) -> int:
    try:
        return int(val)
    except ValueError as err:
        raise _conversion_error(key, val, "int") from err


def validate_positive_float(
    val: str, key: str, strictly_positive: bool = False
) -> float:
    v = validate_float(val, key)
    if v < 0 or (v <= 0 and strictly_positive):
        raise ObservationConfigError.with_context(
            f'Failed to validate "{val}" in {key}={val}.'
            f" {key} must be given a "
            f"{'strictly ' if strictly_positive else ''}positive value.",
            val,
        )
    return v


def validate_localization(val: dict[str, Any], obs_name: str) -> None:
    errors = []
    if "EAST" not in val:
        errors.append(_missing_value_error(f"LOCALIZATION for {obs_name}", "EAST"))
    if "NORTH" not in val:
        errors.append(_missing_value_error(f"LOCALIZATION for {obs_name}", "NORTH"))
    for key in val:
        if key not in {"EAST", "NORTH", "RADIUS"}:
            errors.append(_unknown_key_error(key, f"LOCALIZATION for {obs_name}"))
    if errors:
        raise ObservationConfigError.from_collected(errors)


def extract_localization_values(
    localization_dict: dict[LOCALIZATION_KEYS, float | None], value: Any, key: str
) -> None:
    localization_dict["east"] = validate_float(value["EAST"], key)
    localization_dict["north"] = validate_float(value["NORTH"], key)
    localization_dict["radius"] = (
        validate_float(value["RADIUS"], key) if "RADIUS" in value else None
    )


def validate_positive_int(val: str, key: str) -> int:
    try:
        v = int(val)
    except ValueError as err:
        raise _conversion_error(key, val, "int") from err
    if v < 0:
        raise ObservationConfigError.with_context(
            f'Failed to validate "{val}" in {key}={val}.'
            f" {key} must be given a positive value.",
            val,
        )
    return v


def _missing_value_error(name_token: str, value_key: str) -> ObservationConfigError:
    return ObservationConfigError.with_context(
        f'Missing item "{value_key}" in {name_token}', name_token
    )


def _resolve_path(directory: str, filename: str) -> str:
    if not os.path.isabs(filename):
        filename = os.path.join(directory, filename)
    if not os.path.exists(filename):
        raise ObservationConfigError.with_context(
            f"The following keywords did not resolve to a valid path:\n {filename}",
            filename,
        )
    return filename


def _conversion_error(token: str, value: Any, type_name: str) -> ObservationConfigError:
    return ObservationConfigError.with_context(
        f'Could not convert {value} to {type_name}. Failed to validate "{value}"',
        token,
    )


def _unknown_key_error(key: str, name: str) -> ObservationConfigError:
    raise ObservationConfigError.with_context(f"Unknown {key} in {name}", key)


def _unknown_observation_type_error(obs: ObservationDict) -> ObservationConfigError:
    return ObservationConfigError.with_context(
        f"Unexpected type in observations {obs}", obs["name"]
    )
