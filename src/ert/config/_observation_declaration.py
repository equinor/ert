import os
from collections import Counter
from collections.abc import Sequence
from dataclasses import dataclass
from typing import (
    Any,
    Literal,
)

from .parsing import (
    ErrorInfo,
    ObservationBody,
    ObservationConfigError,
    ObservationStatement,
    ObservationType,
    SimpleHistoryStatement,
)

ErrorModes = Literal["REL", "ABS", "RELMIN"]


@dataclass
class ErrorValues:
    error_mode: ErrorModes
    error: float
    error_min: float


@dataclass
class Segment(ErrorValues):
    start: int
    stop: int


@dataclass
class HistoryValues(ErrorValues):
    key: str
    segment: list[tuple[str, Segment]]


HistoryDeclaration = tuple[str, HistoryValues]


@dataclass
class DateValues:
    days: float | None = None
    hours: float | None = None
    date: str | None = None
    restart: int | None = None


@dataclass
class _SummaryValues:
    value: float
    key: str


@dataclass
class SummaryValues(DateValues, ErrorValues, _SummaryValues):
    pass


SummaryDeclaration = tuple[str, SummaryValues]


@dataclass
class _GenObsValues:
    data: str
    value: float | None = None
    error: float | None = None
    index_list: str | None = None
    index_file: str | None = None
    obs_file: str | None = None


@dataclass
class GenObsValues(DateValues, _GenObsValues):
    pass


GenObsDeclaration = tuple[str, GenObsValues]
Declaration = HistoryDeclaration | SummaryDeclaration | GenObsDeclaration
ConfContent = Sequence[Declaration]


def make_observation_declarations(
    directory: str, statements: Sequence[SimpleHistoryStatement | ObservationStatement]
) -> Sequence[Declaration]:
    """Takes observation statements and returns validated observation declarations.

    Param:
        directory: The name of the directory the observation config is located in.
            Used to disambiguate relative paths.
        inp: The collection of statements to validate.
    """
    result: list[Declaration] = []
    error_list: list[ErrorInfo] = []
    for stat in statements:
        try:
            if stat[0] == ObservationType.HISTORY:
                if len(stat) == 2:
                    result.append(
                        (
                            stat[1],
                            _validate_history_values(stat[1], {}),
                        )
                    )
                if len(stat) == 3:
                    result.append(
                        (
                            stat[1],
                            _validate_history_values(
                                stat[1],
                                stat[2],
                            ),
                        )
                    )
            elif stat[0] == ObservationType.SUMMARY:
                if len(stat) != 3:
                    raise _unknown_declaration_error(stat)
                result.append((stat[1], _validate_summary_values(stat[1], stat[2])))
            elif stat[0] == ObservationType.GENERAL:
                if len(stat) != 3:
                    raise _unknown_declaration_error(stat)
                result.append(
                    (
                        stat[1],
                        _validate_gen_obs_values(directory, stat[1], stat[2]),
                    )
                )
            else:
                raise _unknown_declaration_error(stat)
        except ObservationConfigError as err:
            error_list.extend(err.errors)

    if error_list:
        raise ObservationConfigError.from_collected(error_list)

    _validate_unique_names(result)
    return result


def _validate_unique_names(
    conf_content: Sequence[tuple[str, Any]],
) -> None:
    names_counter = Counter(n for n, _ in conf_content)
    duplicate_names = [n for n, c in names_counter.items() if c > 1]
    errors = [
        ErrorInfo(
            f"Duplicate observation name {n}",
        ).set_context(n)
        for n in duplicate_names
    ]
    if errors:
        raise ObservationConfigError.from_collected(errors)


def _validate_history_values(name_token: str, inp: ObservationBody) -> HistoryValues:
    error_mode: ErrorModes = "RELMIN"
    error = 0.1
    error_min = 0.1
    segment = []
    for key, value in inp.items():
        match key:
            case "ERROR":
                error = validate_positive_float(value, key)
            case "ERROR_MIN":
                error_min = validate_positive_float(value, key)
            case "ERROR_MODE":
                error_mode = validate_error_mode(value)
            case ("SEGMENT", segment_name):
                segment.append(
                    (segment_name, _validate_segment_dict(segment_name, value))
                )
            case _:
                raise _unknown_key_error(str(key), name_token)

    return HistoryValues(
        key=name_token,
        error_mode=error_mode,
        error=error,
        error_min=error_min,
        segment=segment,
    )


def _validate_summary_values(name_token: str, inp: ObservationBody) -> SummaryValues:
    error_mode: ErrorModes = "ABS"
    summary_key = None

    date_dict: DateValues = DateValues()
    float_values: dict[str, float] = {"ERROR_MIN": 0.1}
    for key, value in inp.items():
        match key:
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
                raise _unknown_key_error(str(key), name_token)
    if "VALUE" not in float_values:
        raise _missing_value_error(name_token, "VALUE")
    if summary_key is None:
        raise _missing_value_error(name_token, "KEY")
    if "ERROR" not in float_values:
        raise _missing_value_error(name_token, "ERROR")

    return SummaryValues(
        error_mode=error_mode,
        error=float_values["ERROR"],
        error_min=float_values["ERROR_MIN"],
        key=summary_key,
        value=float_values["VALUE"],
        **date_dict.__dict__,
    )


def _validate_segment_dict(name_token: str, inp: dict[str, Any]) -> Segment:
    start = None
    stop = None
    error_mode: ErrorModes = "RELMIN"
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
        start=start,
        stop=stop,
        error_mode=error_mode,
        error=error,
        error_min=error_min,
    )


def _validate_gen_obs_values(
    directory: str, name_token: str, inp: ObservationBody
) -> GenObsValues:
    try:
        data = inp["DATA"]
    except KeyError as err:
        raise _missing_value_error(name_token, "DATA") from err

    output: GenObsValues = GenObsValues(data=data)
    for key, value in inp.items():
        match key:
            case "RESTART":
                output.restart = validate_positive_int(value, key)
            case "VALUE":
                output.value = validate_float(value, key)
            case "ERROR" | "DAYS" | "HOURS":
                setattr(output, str(key).lower(), validate_positive_float(value, key))
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
                raise _unknown_key_error(str(key), name_token)
    if output.value is not None and output.error is None:
        raise ObservationConfigError.with_context(
            f"For GENERAL_OBSERVATION {name_token}, with"
            f" VALUE = {output.value}, ERROR must also be given.",
            name_token,
        )
    return output


def validate_error_mode(inp: str) -> ErrorModes:
    if inp == "REL":
        return "REL"
    if inp == "ABS":
        return "ABS"
    if inp == "RELMIN":
        return "RELMIN"
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


def _unknown_declaration_error(
    decl: SimpleHistoryStatement | ObservationStatement,
) -> ObservationConfigError:
    return ObservationConfigError.with_context(
        f"Unexpected declaration in observations {decl}", decl[1]
    )
