import os
from enum import Enum, auto
from typing import Any, Dict, List, Literal, Tuple, TypedDict, Union

from lark import Lark, Transformer
from typing_extensions import NotRequired

from .config_errors import ConfigValidationError
from .error_info import ErrorInfo
from .file_context_token import FileContextToken
from .lark_parser import FileContextTransformer

ErrorModes = Literal["REL", "ABS", "RELMIN"]


class SegmentDict(TypedDict):
    START: int
    STOP: int
    ERROR_MODE: ErrorModes
    ERROR: float
    ERROR_MIN: float


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


class HistoryValues(TypedDict):
    ERROR: float
    ERROR_MIN: float
    ERROR_MODE: ErrorModes
    SEGMENT: List[Tuple[str, SegmentDict]]


HistoryDeclaration = Tuple[
    Literal[ObservationType.HISTORY], FileContextToken, HistoryValues
]


class DateDict(TypedDict):
    DAYS: NotRequired[float]
    HOURS: NotRequired[float]
    DATE: NotRequired[str]
    RESTART: NotRequired[int]


class SummaryValues(DateDict):
    VALUE: float
    ERROR: float
    ERROR_MIN: float
    ERROR_MODE: ErrorModes
    KEY: str


SummaryDeclaration = Tuple[
    Literal[ObservationType.SUMMARY], FileContextToken, SummaryValues
]


class GenObsValues(DateDict):
    DATA: str
    VALUE: NotRequired[float]
    ERROR: NotRequired[float]
    INDEX_LIST: NotRequired[str]
    OBS_FILE: NotRequired[str]
    VALUE: NotRequired[float]


GenObsDeclaration = Tuple[
    Literal[ObservationType.GENERAL], FileContextToken, GenObsValues
]
Declaration = Union[HistoryDeclaration, SummaryDeclaration, GenObsDeclaration]
ConfContent = List[Declaration]


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
    return (FileContextTransformer(filename) * TreeToObservations()).transform(
        observations_parser.parse(content)
    )


observations_parser = Lark(
    r"""
    start: observation*
    ?observation: type STRING value? ";"
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

    COMMENT: /--[^\n]*/
    %ignore COMMENT
    """
)


class TreeToObservations(Transformer):
    start = list

    def observation(self, tree):
        return tuple([ObservationType.from_rule(tree[0].data), *tree[1:]])

    def segment(self, tree):
        return ("SEGMENT", tuple(tree))

    object = dict
    pair = tuple


def _validate_conf_content(
    directory: str,
    inp: List[
        Union[
            SimpleHistoryDeclaration,
            Tuple[ObservationType, FileContextToken, Dict[FileContextToken, Any]],
        ]
    ],
):
    result = []
    for decl in inp:
        if decl[0] == ObservationType.HISTORY:
            if len(decl) == 2:
                result.append((decl[0], decl[1], _validate_history_values(decl[1], {})))
            if len(decl) == 3:
                result.append(
                    (decl[0], decl[1], _validate_history_values(decl[1], decl[2]))
                )
        elif decl[0] == ObservationType.SUMMARY:
            if len(decl) != 3:
                raise _unknown_declaration_error(decl)
            result.append(
                (decl[0], decl[1], _validate_summary_values(decl[1], decl[2]))
            )
        elif decl[0] == ObservationType.GENERAL:
            if len(decl) != 3:
                raise _unknown_declaration_error(decl)
            result.append(
                (
                    decl[0],
                    decl[1],
                    _validate_gen_obs_values(directory, decl[1], decl[2]),
                )
            )
        else:
            raise _unknown_declaration_error(decl)
    return result


def _validate_history_values(
    name_token: FileContextToken, inp: Dict[FileContextToken, Any]
) -> HistoryValues:
    error_mode = "RELMIN"
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

    return {
        "ERROR_MODE": error_mode,
        "ERROR": error,
        "ERROR_MIN": error_min,
        "SEGMENT": segment,
    }


def _validate_summary_values(
    name_token: FileContextToken, inp: Dict[FileContextToken, Any]
) -> SummaryValues:
    error_mode = "ABS"
    summary_key = None

    date_dict: DateDict = {}
    float_values: Dict[str, float] = {"ERROR_MIN": 0.1}
    for key, value in inp.items():
        if key == "RESTART":
            date_dict[str(key)] = validate_positive_int(value, key)
        elif key in ["ERROR", "ERROR_MIN"]:
            float_values[str(key)] = validate_positive_float(value, key)
        elif key in ["DAYS", "HOURS"]:
            date_dict[str(key)] = validate_positive_float(value, key)
        elif key == "VALUE":
            float_values[str(key)] = validate_float(value, key)
        elif key == "ERROR_MODE":
            error_mode = validate_error_mode(value)
        elif key == "KEY":
            summary_key = value
        elif key == "DATE":
            date_dict[str(key)] = value
        else:
            raise _unknown_key_error(key, name_token)
    if "VALUE" not in float_values:
        raise _missing_value_error(name_token, "VALUE")
    if summary_key is None:
        raise _missing_value_error(name_token, "KEY")
    if "ERROR" not in float_values:
        raise _missing_value_error(name_token, "ERROR")

    return {
        "ERROR_MODE": error_mode,
        "ERROR": float_values["ERROR"],
        "ERROR_MIN": float_values["ERROR_MIN"],
        "KEY": summary_key,
        "VALUE": float_values["VALUE"],
        **date_dict,
    }


def _validate_segment_dict(
    name_token: FileContextToken, inp: Dict[FileContextToken, Any]
) -> SegmentDict:
    start = None
    stop = None
    error_mode = "RELMIN"
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
    return {
        "START": start,
        "STOP": stop,
        "ERROR_MODE": error_mode,
        "ERROR": error,
        "ERROR_MIN": error_min,
    }


def _validate_gen_obs_values(
    directory: str, name_token: FileContextToken, inp: Dict[FileContextToken, Any]
) -> GenObsValues:
    try:
        output: GenObsValues = {"DATA": inp["DATA"]}
    except KeyError as err:
        raise _missing_value_error(name_token, "DATA") from err

    for key, value in inp.items():
        if key == "RESTART":
            output[str(key)] = validate_positive_int(value, key)
        elif key == "VALUE":
            output[str(key)] = validate_float(value, key)
        elif key in ["ERROR", "ERROR_MIN", "DAYS", "HOURS"]:
            output[str(key)] = validate_positive_float(value, key)
        elif key in ["DATE", "INDEX_LIST"]:
            output[str(key)] = value
        elif key == "OBS_FILE":
            filename = value
            if not os.path.isabs(filename):
                filename = os.path.join(directory, filename)
            if not os.path.exists(filename):
                raise ObservationConfigError(
                    [
                        ErrorInfo(
                            "The following keywords did not"
                            " resolve to a valid path:\n OBS_FILE",
                            key.filename,
                        ).set_context(value)
                    ]
                )
            output[str(key)] = filename
        elif key == "DATA":
            output[str(key)] = value
        else:
            raise _unknown_key_error(key, name_token)
    return output


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


def validate_error_mode(inp: FileContextToken) -> ErrorModes:
    inp_str = str(inp)
    if inp_str == "REL":
        return inp_str
    if inp_str == "ABS":
        return inp_str
    if inp_str == "RELMIN":
        return inp_str
    raise ObservationConfigError(
        [
            ErrorInfo(
                f'Unexpected ERROR_MODE {inp}. Failed to validate "{inp}"', inp.filename
            ).set_context(inp)
        ]
    )


def validate_float(val: str, key: FileContextToken):
    try:
        return float(val)
    except ValueError as err:
        raise _conversion_error(key, val, "float") from err


def validate_int(val: str, key: FileContextToken):
    try:
        return int(val)
    except ValueError as err:
        raise _conversion_error(key, val, "int") from err


def validate_positive_float(val: str, key: FileContextToken):
    v = validate_float(val, key)
    if v < 0:
        raise ObservationConfigError(
            [
                ErrorInfo(
                    f'Failed to validate "{val}" in {key}={val}.'
                    f" {key} must be given a positive value.",
                    key.filename,
                ).set_context(val)
            ]
        )
    return v


def validate_positive_int(val: str, key: FileContextToken):
    try:
        v = int(val)
    except ValueError as err:
        raise _conversion_error(key, val, "int") from err
    if v < 0:
        raise ObservationConfigError(
            [
                ErrorInfo(
                    f'Failed to validate "{val}" in {key}={val}.'
                    f" {key} must be given a positive value.",
                    key.filename,
                ).set_context(val)
            ]
        )
    return v


def _missing_value_error(
    name_token: FileContextToken, value_key: str
) -> ObservationConfigError:
    return ObservationConfigError(
        [
            ErrorInfo(
                f'Missing item "{value_key}" in {name_token}', name_token.filename
            ).set_context(name_token)
        ]
    )


def _conversion_error(
    token: FileContextToken, value: Any, type_name: str
) -> ObservationConfigError:
    return ObservationConfigError(
        [
            ErrorInfo(
                f"Could not convert {value} to "
                f'{type_name}. Failed to validate "{value}"',
                token.filename,
            ).set_context(token)
        ]
    )


def _unknown_key_error(key: FileContextToken, name: str) -> ObservationConfigError:
    raise ObservationConfigError(
        [ErrorInfo(f"Unknown {key} in {name}", key.filename).set_context(key)]
    )


def _unknown_declaration_error(
    decl: Union[
        SimpleHistoryDeclaration, Tuple[ObservationType, FileContextToken, Any]
    ],
):
    return ObservationConfigError(
        [
            ErrorInfo(
                f"Unexpected declaration in observations {decl}",
                decl[1].filename,
            ).set_context(decl[1])
        ]
    )
