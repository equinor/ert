from enum import StrEnum
from typing import Any, assert_never, no_type_check

from lark import Lark, Token, Transformer, UnexpectedCharacters, UnexpectedToken
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
    RFT = "RFT_OBSERVATION"


ObservationDict = dict[str, Any]


def parse_observations(content: str, filename: str) -> list[ObservationDict]:
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
        line: int | None = e.line
        end_line: int | None = e.line
        column: int | None = e.column
        end_column: int | None = e.token.end_column
        match e, sorted(e.expected):
            case UnexpectedToken(
                token=unexpected_token,
                token_history=[
                    Token(
                        type="PARAMETER_NAME",
                        value=property_name,
                        line=line,
                        column=column,
                        end_line=end_line,
                        end_column=end_column,
                    )
                ],
            ), ["EQUAL"]:
                message = (
                    f"Expected assignment to property '{property_name}'. "
                    f"Got '{unexpected_token}' instead."
                )
            case UnexpectedToken(
                token=unexpected_token,
                token_history=[
                    Token(
                        type="OBSERVATION_NAME",
                    )
                ],
            ), ["LBRACE", "SEMICOLON"]:
                message = (
                    "Expected either start of observation body ('{') "
                    f"or end of observation (';'), got '{unexpected_token}' instead."
                )
            case UnexpectedToken(
                token=unexpected_token,
            ), ["TYPE"]:
                message = (
                    f"Unknown observation type '{unexpected_token}', "
                    f"expected either 'RFT_OBSERVATION', 'GENERAL_OBSERVATION', "
                    f"'SUMMARY_OBSERVATION' or 'HISTORY_OBSERVATION'."
                )
            case UnexpectedToken(token=unexpected_char, expected=allowed_chars), _:
                unexpected_line = content.splitlines()[e.line - 1]
                message = (
                    f"Observation parsing failed: "
                    f"Did not expect character: {unexpected_char}"
                    f" (on line {e.line}: {unexpected_line}). "
                    f"Expected one of {allowed_chars}."
                )
            case default, _:
                assert_never(default)

        raise ObservationConfigError.from_info(
            ErrorInfo(
                filename=filename,
                message=message,
                line=line,
                end_line=end_line,
                column=column,
                end_column=end_column,
            )
        ) from e


observations_parser = Lark(
    r"""
    start: observation*
    ?observation: type OBSERVATION_NAME object? ";"
    TYPE: "HISTORY_OBSERVATION"
      | "SUMMARY_OBSERVATION"
      | "GENERAL_OBSERVATION"
      | "RFT_OBSERVATION"
    type: TYPE
    ?value: object
          | STRING


    CHAR: /[^; \t\n{}=]/
    STRING : CHAR+
    OBSERVATION_NAME : CHAR+
    PARAMETER_NAME : CHAR+
    object : "{" [(declaration";")*] "}"
    ?declaration: "SEGMENT" STRING object -> segment
                | "LOCALIZATION" object -> localization
                | pair
    pair   : PARAMETER_NAME "=" value


    %import common.WS
    %ignore WS

    COMMENT.9: "--" /[^\n]/*
    %ignore COMMENT
    """,
    parser="lalr",
)


class TreeToObservations(Transformer[FileContextToken, list[ObservationDict]]):
    start = list

    @staticmethod
    @no_type_check
    def observation(tree):
        if len(tree) == 2:
            return {
                "type": ObservationType(tree[0].children[0]),
                "name": tree[1],
            }
        else:
            non_segments = {
                k: v for k, v in tree[2].items() if not isinstance(k, tuple)
            }
            segments = [(k[1], v) for k, v in tree[2].items() if isinstance(k, tuple)]
            error_list = []
            for unknown_key in ["type", "segments", "name"]:
                if unknown_key in non_segments:
                    error_list.append(
                        ErrorInfo(f"Unknown {unknown_key} in {tree[1]}").set_context(
                            tree[1]
                        )
                    )
            if error_list:
                raise ObservationConfigError.from_collected(error_list)

            res = {
                "type": ObservationType(tree[0].children[0]),
                "name": tree[1],
                **non_segments,
            }
            if segments:
                res["segments"] = segments
            return res

    @staticmethod
    @no_type_check
    def segment(tree):
        return (("SEGMENT", tree[0]), tree[1])

    @staticmethod
    @no_type_check
    def localization(tree):
        return ("LOCALIZATION", tree[0])

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
