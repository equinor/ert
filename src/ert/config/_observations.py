from __future__ import annotations

import logging
import os
from collections.abc import Sequence
from datetime import datetime
from enum import StrEnum
from typing import Annotated, Any, Literal, Self

import pandas as pd
from pydantic import BaseModel, Field

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
    key: str  #: The :term:`summary key` in the summary response
    date: str
    location_x: float | None = None
    location_y: float | None = None
    location_range: float | None = None


class ObservationError(BaseModel):
    error_mode: ErrorModes
    error: float
    error_min: float


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


class SummaryObservation(_SummaryValues, ObservationError):
    @classmethod
    def from_obs_dict(
        cls, directory: str, observation_dict: ObservationDict
    ) -> list[Self]:
        error_mode = ErrorModes.ABS
        summary_key = None

        date: str | None = None
        float_values: dict[str, float] = {"ERROR_MIN": 0.1}
        localization_values: dict[str, float | None] = {}
        for key, value in observation_dict.items():
            match key:
                case "type" | "name":
                    pass
                case "ERROR" | "ERROR_MIN":
                    float_values[str(key)] = validate_positive_float(value, key)
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
                    localization_values["x"] = validate_float(value["EAST"], key)
                    localization_values["y"] = validate_float(value["NORTH"], key)
                    localization_values["range"] = (
                        validate_float(value["RADIUS"], key)
                        if "RADIUS" in value
                        else None
                    )
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
        instance = cls(
            name=observation_dict["name"],
            error_mode=error_mode,
            error=float_values["ERROR"],
            error_min=float_values["ERROR_MIN"],
            key=summary_key,
            value=float_values["VALUE"],
            location_x=localization_values.get("x"),
            location_y=localization_values.get("y"),
            location_range=localization_values.get("range"),
            date=standardized_date,
        )
        # Bypass pydantic discarding context
        # only relevant for ERT config surfacing validation errors
        # irrelevant for runmodels etc.
        instance.name = observation_dict["name"]

        return [instance]


class _GeneralObservation(BaseModel):
    type: Literal["general_observation"] = "general_observation"
    name: str
    data: str
    value: float | None = None
    error: float | None = None
    index_list: str | None = None
    index_file: str | None = None
    obs_file: str | None = None
    restart: int | None = None


class GeneralObservation(_GeneralObservation):
    @classmethod
    def from_obs_dict(
        cls, directory: str, observation_dict: ObservationDict
    ) -> list[Self]:
        try:
            data = observation_dict["DATA"]
        except KeyError as err:
            raise _missing_value_error(observation_dict["name"], "DATA") from err

        output = cls(name=observation_dict["name"], data=data)
        for key, value in observation_dict.items():
            match key:
                case "type" | "name":
                    pass
                case "RESTART":
                    output.restart = validate_positive_int(value, key)
                case "VALUE":
                    output.value = validate_float(value, key)
                case "ERROR":
                    setattr(
                        output, str(key).lower(), validate_positive_float(value, key)
                    )
                case "DATE" | "DAYS" | "HOURS":
                    raise ObservationConfigError.with_context(
                        (
                            "GENERAL_OBSERVATION must use RESTART to specify "
                            "report step. Please run:\n ert convert_observations "
                            "<your_ert_config.ert>\nto migrate the observation config "
                            "to use the correct format."
                        ),
                        key,
                    )
                case "INDEX_LIST":
                    output.index_list = value
                case "OBS_FILE" | "INDEX_FILE":
                    assert not isinstance(key, tuple)
                    filename = value
                    if not os.path.isabs(filename):
                        filename = os.path.join(directory, filename)
                    if not os.path.exists(filename):
                        raise ObservationConfigError.with_context(
                            "The following keywords did not"
                            f" resolve to a valid path:\n {key}",
                            value,
                        )
                    setattr(output, str(key).lower(), filename)
                case "DATA":
                    output.data = value
                case _:
                    raise _unknown_key_error(str(key), observation_dict["name"])
        if output.value is not None and output.error is None:
            raise ObservationConfigError.with_context(
                f"For GENERAL_OBSERVATION {observation_dict['name']}, with"
                f" VALUE = {output.value}, ERROR must also be given.",
                observation_dict["name"],
            )

        # Bypass pydantic discarding context
        # only relevant for ERT config surfacing validation errors
        # irrelevant for runmodels etc.
        output.name = observation_dict["name"]

        return [output]


class RFTObservation(BaseModel):
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

    @classmethod
    def from_csv(
        cls,
        directory: str,
        observation_dict: ObservationDict,
        filename: str,
        observed_property: str = "PRESSURE",
    ) -> list[Self]:
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

        return [
            cls(
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
            )
            for row in csv_file.itertuples(index=True)
        ]

    @classmethod
    def from_obs_dict(
        cls, directory: str, observation_dict: ObservationDict
    ) -> list[Self]:
        csv_filename = None
        well = None
        observed_property = None
        observed_value = None
        error = None
        date = None
        north = None
        east = None
        tvd = None
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

        instance = cls(
            name=observation_dict["name"],
            well=well,
            property=observed_property,
            value=observed_value,
            error=error,
            date=date,
            north=north,
            east=east,
            tvd=tvd,
        )

        # Bypass pydantic discarding context
        # only relevant for ERT config surfacing validation errors
        # irrelevant for runmodels etc.
        instance.name = observation_dict["name"]

        return [instance]


Observation = Annotated[
    (SummaryObservation | GeneralObservation | RFTObservation),
    Field(discriminator="type"),
]

_TYPE_TO_CLASS: dict[ObservationType, type[Observation]] = {
    ObservationType.SUMMARY: SummaryObservation,
    ObservationType.GENERAL: GeneralObservation,
    ObservationType.RFT: RFTObservation,
}


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


def validate_positive_float(val: str, key: str) -> float:
    v = validate_float(val, key)
    if v < 0:
        raise ObservationConfigError.with_context(
            f'Failed to validate "{val}" in {key}={val}.'
            f" {key} must be given a positive value.",
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
