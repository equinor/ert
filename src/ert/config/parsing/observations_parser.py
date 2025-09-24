from enum import StrEnum
from typing import (
    Any,
    no_type_check,
)

from lark import Lark, Transformer, UnexpectedCharacters, UnexpectedToken
from lark.exceptions import VisitError

from ._file_context_transformer import FileContextTransformer
from .config_errors import ConfigValidationError
from .error_info import ErrorInfo
from .file_context_token import FileContextToken


class ObservationConfigError(ConfigValidationError):
    pass


class ObservationType(StrEnum):
    HISTORY = "HISTORY_OBSERVATION"
    SUMMARY = "SUMMARY_OBSERVATION"
    GENERAL = "GENERAL_OBSERVATION"


ObservationBody = dict[str | tuple[str, str], Any]
ObservationName = str
ObservationStatement = tuple[ObservationType, ObservationName, ObservationBody]


def parse_observations(content: str, filename: str) -> list[ObservationStatement]:
    try:
        return (FileContextTransformer(filename) * TreeToObservations()).transform(
            observations_parser.parse(content)
        )
    except VisitError as err:
        if isinstance(err.orig_exc, ObservationConfigError):
            raise err.orig_exc from None
        raise err from None
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
    TYPE: "HISTORY_OBSERVATION" | "SUMMARY_OBSERVATION" | "GENERAL_OBSERVATION"
    type: TYPE
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


class TreeToObservations(Transformer[FileContextToken, list[ObservationStatement]]):
    start = list

    @staticmethod
    @no_type_check
    def observation(tree):
        if len(tree) == 2:
            return (ObservationType(tree[0].children[0]), *tree[1:], {})
        else:
            return (ObservationType(tree[0].children[0]), *tree[1:])

    @staticmethod
    @no_type_check
    def segment(tree):
        return (("SEGMENT", tree[0]), tree[1])

    @staticmethod
    @no_type_check
    def object(tree):
        keys = set()
        error_list: list[ErrorInfo] = []
        for key, *_ in tree:
            if key in keys:
                error_list.append(
                    ErrorInfo(f"Observation contains duplicate key {key}").set_context(
                        key
                    )
                )
            keys.add(key)
        if error_list:
            raise ObservationConfigError.from_collected(error_list)
        return dict(tree)

    pair = tuple
