from enum import StrEnum
from typing import (
    Any,
    Literal,
    no_type_check,
)

from lark import Lark, Transformer, UnexpectedCharacters, UnexpectedToken

from ._file_context_transformer import FileContextTransformer
from .config_errors import ConfigValidationError
from .error_info import ErrorInfo
from .file_context_token import FileContextToken


class ObservationConfigError(ConfigValidationError):
    pass


class ObservationType(StrEnum):
    HISTORY = "HISTORY"
    SUMMARY = "SUMMARY"
    GENERAL = "GENERAL"

    @classmethod
    def from_rule(cls, rule: str) -> "ObservationType":
        if rule == "summary":
            return cls.SUMMARY
        if rule == "general":
            return cls.GENERAL
        if rule == "history":
            return cls.HISTORY
        raise ValueError(f"Unexpected observation type {rule}")


SimpleHistoryDeclaration = tuple[Literal[ObservationType.HISTORY], FileContextToken]


def parse_observations(
    content: str, filename: str
) -> list[
    SimpleHistoryDeclaration
    | tuple[ObservationType, FileContextToken, dict[FileContextToken, Any]]
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
        list[
            SimpleHistoryDeclaration
            | tuple[ObservationType, FileContextToken, dict[FileContextToken, Any]]
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
