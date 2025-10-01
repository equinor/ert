import os
from collections import Counter
from collections.abc import Sequence
from dataclasses import dataclass
from enum import StrEnum
from itertools import starmap
from typing import Any, Self

from .parsing import (
    ErrorInfo,
    ObservationConfigError,
    ObservationDict,
    ObservationType,
)


class ErrorModes(StrEnum):
    REL = "REL"
    ABS = "ABS"
    RELMIN = "RELMIN"


@dataclass
class ObservationError:
    error_mode: ErrorModes
    error: float
    error_min: float


@dataclass
class Segment(ObservationError):
    name: str
    start: int
    stop: int


@dataclass
class HistoryObservation(ObservationError):
    name: str
    segments: list[Segment]

    @property
    def key(self) -> str:
        """The :term:`summary key` to be fetched from :ref:`refcase`."""
        # For history observations the key is also the name, ie.
        # "HISTORY_OBSERVATION FOPR" means to add the values from
        # the summary vector FOPRH in refcase as observations.
        return self.name

    @classmethod
    def from_obs_dict(cls, directory: str, observation_dict: ObservationDict) -> Self:
        error_mode = ErrorModes.RELMIN
        error = 0.1
        error_min = 0.1
        segments = []
        for key, value in observation_dict.items():
            match key:
                case "type" | "name":
                    pass
                case "ERROR":
                    error = validate_positive_float(value, key)
                case "ERROR_MIN":
                    error_min = validate_positive_float(value, key)
                case "ERROR_MODE":
                    error_mode = validate_error_mode(value)
                case "segments":
                    segments = list(starmap(_validate_segment_dict, value))
                case _:
                    raise _unknown_key_error(str(key), observation_dict["name"])

        return cls(
            name=observation_dict["name"],
            error_mode=error_mode,
            error=error,
            error_min=error_min,
            segments=segments,
        )


@dataclass
class ObservationDate:
    days: float | None = None
    hours: float | None = None
    date: str | None = None
    restart: int | None = None


@dataclass
class _SummaryValues:
    name: str
    value: float
    key: str  #: The :term:`summary key` in the summary response


@dataclass
class SummaryObservation(ObservationDate, ObservationError, _SummaryValues):
    @classmethod
    def from_obs_dict(cls, directory: str, observation_dict: ObservationDict) -> Self:
        error_mode = ErrorModes.ABS
        summary_key = None

        date_dict: ObservationDate = ObservationDate()
        float_values: dict[str, float] = {"ERROR_MIN": 0.1}
        for key, value in observation_dict.items():
            match key:
                case "type" | "name":
                    pass
                case "RESTART":
                    date_dict.restart = validate_positive_int(value, key)
                case "ERROR" | "ERROR_MIN":
                    float_values[str(key)] = validate_positive_float(value, key)
                case "DAYS" | "HOURS":
                    setattr(
                        date_dict, str(key).lower(), validate_positive_float(value, key)
                    )
                case "VALUE":
                    float_values[str(key)] = validate_float(value, key)
                case "ERROR_MODE":
                    error_mode = validate_error_mode(value)
                case "KEY":
                    summary_key = value
                case "DATE":
                    date_dict.date = value
                case _:
                    raise _unknown_key_error(str(key), observation_dict["name"])
        if "VALUE" not in float_values:
            raise _missing_value_error(observation_dict["name"], "VALUE")
        if summary_key is None:
            raise _missing_value_error(observation_dict["name"], "KEY")
        if "ERROR" not in float_values:
            raise _missing_value_error(observation_dict["name"], "ERROR")

        return cls(
            name=observation_dict["name"],
            error_mode=error_mode,
            error=float_values["ERROR"],
            error_min=float_values["ERROR_MIN"],
            key=summary_key,
            value=float_values["VALUE"],
            **date_dict.__dict__,
        )


@dataclass
class _GeneralObservation:
    name: str
    data: str
    value: float | None = None
    error: float | None = None
    index_list: str | None = None
    index_file: str | None = None
    obs_file: str | None = None


@dataclass
class GeneralObservation(ObservationDate, _GeneralObservation):
    @classmethod
    def from_obs_dict(cls, directory: str, observation_dict: ObservationDict) -> Self:
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
                case "ERROR" | "DAYS" | "HOURS":
                    setattr(
                        output, str(key).lower(), validate_positive_float(value, key)
                    )
                case "DATE" | "INDEX_LIST":
                    setattr(output, str(key).lower(), value)
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
        return output


Observation = HistoryObservation | SummaryObservation | GeneralObservation

_TYPE_TO_CLASS: dict[ObservationType, type[Observation]] = {
    ObservationType.HISTORY: HistoryObservation,
    ObservationType.SUMMARY: SummaryObservation,
    ObservationType.GENERAL: GeneralObservation,
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
    error_list: list[ErrorInfo] = []
    for obs_dict in observation_dicts:
        try:
            result.append(
                _TYPE_TO_CLASS[obs_dict["type"]].from_obs_dict(directory, obs_dict)
            )
        except KeyError as err:
            raise _unknown_observation_type_error(obs_dict) from err
        except ObservationConfigError as err:
            error_list.extend(err.errors)

    if error_list:
        raise ObservationConfigError.from_collected(error_list)

    _validate_unique_names(result)
    return result


def _validate_unique_names(
    observations: Sequence[Observation],
) -> None:
    names_counter = Counter(d.name for d in observations)
    duplicate_names = [n for n, c in names_counter.items() if c > 1]
    errors = [
        ErrorInfo(
            f"Duplicate observation name {n}",
        ).set_context(n)
        for n in duplicate_names
    ]
    if errors:
        raise ObservationConfigError.from_collected(errors)


def _validate_segment_dict(name_token: str, inp: dict[str, Any]) -> Segment:
    start = None
    stop = None
    error_mode = ErrorModes.RELMIN
    error = 0.1
    error_min = 0.1
    for key, value in inp.items():
        match key:
            case "START":
                start = validate_int(value, key)
            case "STOP":
                stop = validate_int(value, key)
            case "ERROR":
                error = validate_positive_float(value, key)
            case "ERROR_MIN":
                error_min = validate_positive_float(value, key)
            case "ERROR_MODE":
                error_mode = validate_error_mode(value)
            case _:
                raise _unknown_key_error(key, name_token)

    if start is None:
        raise _missing_value_error(name_token, "START")
    if stop is None:
        raise _missing_value_error(name_token, "STOP")
    return Segment(
        name=name_token,
        start=start,
        stop=stop,
        error_mode=error_mode,
        error=error,
        error_min=error_min,
    )


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
