import os
from collections import Counter
from dataclasses import dataclass
from enum import Enum, auto
from typing import (
    Any,
    Dict,
    List,
    Literal,
    Optional,
    Sequence,
    Tuple,
    Union,
    cast,
    no_type_check,
)

from lark import Lark, Transformer, UnexpectedCharacters, UnexpectedToken

from .config_errors import ConfigValidationError
from .error_info import ErrorInfo
from .file_context_token import FileContextToken
from .lark_parser import FileContextTransformer

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


class ObservationType(Enum):
    HISTORY = auto()
    SUMMARY = auto()
    GENERAL = auto()

    @classmethod
    def from_rule(cls, rule: str) -> "ObservationType":
        if rule == "summary":
            return cls.SUMMARY
        if rule == "general":
            return cls.GENERAL
        if rule == "history":
            return cls.HISTORY
        raise ValueError(f"Unexpected observation type {rule}")


SimpleHistoryDeclaration = Tuple[Literal[ObservationType.HISTORY], FileContextToken]


@dataclass
class HistoryValues(ErrorValues):
    key: str
    segment: List[Tuple[str, Segment]]


HistoryDeclaration = Tuple[FileContextToken, HistoryValues]


@dataclass
class DateValues:
    days: Optional[float] = None
    hours: Optional[float] = None
    date: Optional[str] = None
    restart: Optional[int] = None


@dataclass
class _SummaryValues:
    value: float
    key: str


@dataclass
class SummaryValues(DateValues, ErrorValues, _SummaryValues):
    pass


SummaryDeclaration = Tuple[FileContextToken, SummaryValues]


@dataclass
class _GenObsValues:
    data: str
    value: Optional[float] = None
    error: Optional[float] = None
    index_list: Optional[str] = None
    index_file: Optional[str] = None
    obs_file: Optional[str] = None


@dataclass
class GenObsValues(DateValues, _GenObsValues):
    pass


GenObsDeclaration = Tuple[FileContextToken, GenObsValues]
Declaration = Union[HistoryDeclaration, SummaryDeclaration, GenObsDeclaration]
ConfContent = Sequence[Declaration]


def parse(filename: str) -> ConfContent:
    filepath = os.path.normpath(os.path.abspath(filename))
    with open(filepath, encoding="utf-8") as f:
        return _validate_conf_content(
            os.path.dirname(filename), _parse_content(f.read(), filename)
        )


def _parse_content(
    content: str, filename: str
) -> List[
    Union[
        SimpleHistoryDeclaration,
        Tuple[ObservationType, FileContextToken, Dict[FileContextToken, Any]],
    ]
]:
    try:
        return (FileContextTransformer(filename) * TreeToObservations()).transform(
            observations_parser.parse(content)
        )
    except UnexpectedCharacters as e:
        unexpected_char = e.char
        allowed_chars = e.allowed
        unexpected_line = content.splitlines()[e.line - 1]
        message = (
            f"Observation parsing failed: Did not expect character: {unexpected_char}"
            f" (on line {e.line}: {unexpected_line}). "
            f"Expected one of {allowed_chars}."
        )

        raise ObservationConfigError.from_info(
            ErrorInfo(
                filename=filename,
                message=message,
                line=e.line,
                end_line=e.line,
                column=e.column,
                end_column=e.column + 1,
            )
        ) from e
    except UnexpectedToken as e:
        unexpected_char = e.token
        allowed_chars = e.expected
        unexpected_line = content.splitlines()[e.line - 1]
        message = (
            f"Observation parsing failed: Did not expect character: {unexpected_char}"
            f" (on line {e.line}: {unexpected_line}). "
            f"Expected one of {allowed_chars}."
        )

        raise ObservationConfigError.from_info(
            ErrorInfo(
                filename=filename,
                message=message,
                line=e.line,
                end_line=e.line,
                column=e.column,
                end_column=e.column + 1,
            )
        ) from e


observations_parser = Lark(
    r"""
    start: observation*
    ?observation: type STRING object? ";"
    type: "HISTORY_OBSERVATION" -> history
        | "SUMMARY_OBSERVATION" -> summary
        | "GENERAL_OBSERVATION" -> general
    ?value: object
          | STRING


    CHAR: /[^; \t\n{}=]/
    STRING : CHAR+
    object : "{" [(declaration";")*] "}"
    ?declaration: "SEGMENT" STRING object -> segment
                | pair
    pair   : STRING "=" value


    %import common.WS
    %ignore WS

    COMMENT.9: "--" /[^\n]/*
    %ignore COMMENT
    """,
    parser="lalr",
)


class TreeToObservations(
    Transformer[
        FileContextToken,
        List[
            Union[
                SimpleHistoryDeclaration,
                Tuple[ObservationType, FileContextToken, Dict[FileContextToken, Any]],
            ]
        ],
    ]
):
    start = list

    @staticmethod
    @no_type_check
    def observation(tree):
        return (ObservationType.from_rule(tree[0].data), *tree[1:])

    @staticmethod
    @no_type_check
    def segment(tree):
        return ("SEGMENT", tuple(tree))

    object = dict
    pair = tuple


def _validate_conf_content(
    directory: str,
    inp: Sequence[
        Union[
            SimpleHistoryDeclaration,
            Tuple[ObservationType, FileContextToken, Dict[FileContextToken, Any]],
        ]
    ],
) -> ConfContent:
    result: List[Declaration] = []
    error_list: List[ErrorInfo] = []
    for decl in inp:
        try:
            if decl[0] == ObservationType.HISTORY:
                if len(decl) == 2:
                    result.append(
                        (
                            decl[1],
                            _validate_history_values(decl[1], {}),
                        )
                    )
                if len(decl) == 3:
                    result.append(
                        (
                            decl[1],
                            _validate_history_values(
                                decl[1],
                                decl[2],
                            ),
                        )
                    )
            elif decl[0] == ObservationType.SUMMARY:
                if len(decl) != 3:
                    raise _unknown_declaration_error(decl)
                result.append((decl[1], _validate_summary_values(decl[1], decl[2])))
            elif decl[0] == ObservationType.GENERAL:
                if len(decl) != 3:
                    raise _unknown_declaration_error(decl)
                result.append(
                    (
                        decl[1],
                        _validate_gen_obs_values(directory, decl[1], decl[2]),
                    )
                )
            else:
                raise _unknown_declaration_error(decl)
        except ObservationConfigError as err:
            error_list.extend(err.errors)

    if error_list:
        raise ObservationConfigError.from_collected(error_list)

    _validate_unique_names(result)
    return result


def _validate_unique_names(
    conf_content: Sequence[Tuple[FileContextToken, Any]],
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


def _validate_history_values(
    name_token: FileContextToken, inp: Dict[FileContextToken, Any]
) -> HistoryValues:
    error_mode: ErrorModes = "RELMIN"
    error = 0.1
    error_min = 0.1
    segment = []
    for key, value in inp.items():
        if key == "ERROR":
            error = validate_positive_float(value, key)
        elif key == "ERROR_MIN":
            error_min = validate_positive_float(value, key)
        elif key == "ERROR_MODE":
            error_mode = validate_error_mode(value)
        elif key == "SEGMENT":
            segment.append((value[0], _validate_segment_dict(key, value[1])))
        else:
            raise _unknown_key_error(key, name_token)

    return HistoryValues(
        key=name_token,
        error_mode=error_mode,
        error=error,
        error_min=error_min,
        segment=segment,
    )


def _validate_summary_values(
    name_token: FileContextToken, inp: Dict[FileContextToken, Any]
) -> SummaryValues:
    error_mode: ErrorModes = "ABS"
    summary_key = None

    date_dict: DateValues = DateValues()
    float_values: Dict[str, float] = {"ERROR_MIN": 0.1}
    for key, value in inp.items():
        if key == "RESTART":
            date_dict.restart = validate_positive_int(value, key)
        elif key in ["ERROR", "ERROR_MIN"]:
            float_values[str(key)] = validate_positive_float(value, key)
        elif key in ["DAYS", "HOURS"]:
            setattr(date_dict, str(key).lower(), validate_positive_float(value, key))
        elif key == "VALUE":
            float_values[str(key)] = validate_float(value, key)
        elif key == "ERROR_MODE":
            error_mode = validate_error_mode(value)
        elif key == "KEY":
            summary_key = value
        elif key == "DATE":
            date_dict.date = value
        else:
            raise _unknown_key_error(key, name_token)
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


def _validate_segment_dict(
    name_token: FileContextToken, inp: Dict[FileContextToken, Any]
) -> Segment:
    start = None
    stop = None
    error_mode: ErrorModes = "RELMIN"
    error = 0.1
    error_min = 0.1
    for key, value in inp.items():
        if key == "START":
            start = validate_int(value, key)
        elif key == "STOP":
            stop = validate_int(value, key)
        elif key == "ERROR":
            error = validate_positive_float(value, key)
        elif key == "ERROR_MIN":
            error_min = validate_positive_float(value, key)
        elif key == "ERROR_MODE":
            error_mode = validate_error_mode(value)
        else:
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
    directory: str, name_token: FileContextToken, inp: Dict[FileContextToken, Any]
) -> GenObsValues:
    try:
        data = inp[cast(FileContextToken, "DATA")]
    except KeyError as err:
        raise _missing_value_error(name_token, "DATA") from err

    output: GenObsValues = GenObsValues(data=data)
    for key, value in inp.items():
        if key == "RESTART":
            output.restart = validate_positive_int(value, key)
        elif key == "VALUE":
            output.value = validate_float(value, key)
        elif key in ["ERROR", "DAYS", "HOURS"]:
            setattr(output, str(key).lower(), validate_positive_float(value, key))
        elif key in ["DATE", "INDEX_LIST"]:
            setattr(output, str(key).lower(), value)
        elif key in ["OBS_FILE", "INDEX_FILE"]:
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
        elif key == "DATA":
            output.data = value
        else:
            raise _unknown_key_error(key, name_token)
    if output.value is not None and output.error is None:
        raise ObservationConfigError.with_context(
            f"For GENERAL_OBSERVATION {name_token}, with"
            f" VALUE = {output.value}, ERROR must also be given.",
            name_token,
        )
    return output


class ObservationConfigError(ConfigValidationError):
    pass


def validate_error_mode(inp: FileContextToken) -> ErrorModes:
    if inp == "REL":
        return "REL"
    if inp == "ABS":
        return "ABS"
    if inp == "RELMIN":
        return "RELMIN"
    raise ObservationConfigError.with_context(
        f'Unexpected ERROR_MODE {inp}. Failed to validate "{inp}"', inp
    )


def validate_float(val: str, key: FileContextToken) -> float:
    try:
        return float(val)
    except ValueError as err:
        raise _conversion_error(key, val, "float") from err


def validate_int(val: str, key: FileContextToken) -> int:
    try:
        return int(val)
    except ValueError as err:
        raise _conversion_error(key, val, "int") from err


def validate_positive_float(val: str, key: FileContextToken) -> float:
    v = validate_float(val, key)
    if v < 0:
        raise ObservationConfigError.with_context(
            f'Failed to validate "{val}" in {key}={val}.'
            f" {key} must be given a positive value.",
            val,
        )
    return v


def validate_positive_int(val: str, key: FileContextToken) -> int:
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


def _missing_value_error(
    name_token: FileContextToken, value_key: str
) -> ObservationConfigError:
    return ObservationConfigError.with_context(
        f'Missing item "{value_key}" in {name_token}', name_token
    )


def _conversion_error(
    token: FileContextToken, value: Any, type_name: str
) -> ObservationConfigError:
    return ObservationConfigError.with_context(
        f"Could not convert {value} to " f'{type_name}. Failed to validate "{value}"',
        token,
    )


def _unknown_key_error(key: FileContextToken, name: str) -> ObservationConfigError:
    raise ObservationConfigError.with_context(f"Unknown {key} in {name}", key)


def _unknown_declaration_error(
    decl: Union[
        SimpleHistoryDeclaration, Tuple[ObservationType, FileContextToken, Any]
    ],
) -> ObservationConfigError:
    return ObservationConfigError.with_context(
        f"Unexpected declaration in observations {decl}", decl[1]
    )
