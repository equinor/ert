from dataclasses import InitVar, dataclass
from typing import List, Optional, Union

from .lark_parser_file_context_token import FileContextToken
from .lark_parser_types import MaybeWithContext


@dataclass
# pylint: disable=too-many-instance-attributes
class ErrorInfo:
    message: str
    filename: Optional[str]
    start_pos: Optional[int] = None
    line: Optional[int] = None
    column: Optional[int] = None
    end_line: Optional[int] = None
    end_column: Optional[int] = None
    end_pos: Optional[int] = None
    originates_from: InitVar[Union[MaybeWithContext, FileContextToken]] = None
    originates_from_these: InitVar[
        List[Union[MaybeWithContext, FileContextToken]]
    ] = None
    originates_from_keyword: InitVar[Union[MaybeWithContext, FileContextToken]] = None

    def __post_init__(
        self,
        originates_from: Optional[Union[MaybeWithContext, FileContextToken]],
        originates_from_these: Optional[
            List[Union[MaybeWithContext, FileContextToken]]
        ],
        originates_from_keyword: Optional[Union[MaybeWithContext, FileContextToken]],
    ):
        token = None

        def take(origin: Union[MaybeWithContext, FileContextToken], attr: str):
            if isinstance(origin, FileContextToken):
                return origin
            elif hasattr(origin, attr):
                return getattr(origin, attr)

            return None

        if originates_from_keyword is not None:
            token = take(origin=originates_from_keyword, attr="keyword_token")

        elif originates_from is not None:
            token = take(origin=originates_from, attr="token")

        elif originates_from_these is not None:
            tokens = []
            for origin_token in originates_from_these:
                the_token = take(origin=origin_token, attr="token")
                if the_token is not None:
                    tokens.append(the_token)

            if len(tokens) > 0:
                token = FileContextToken.join_tokens(tokens)

        if token is not None:
            self.attach_to_token(token)
        pass

    @property
    def message_with_location(self):
        return (
            self.message
            + f" at line {self.line}, column {self.column}-{self.end_column}"
        )

    @classmethod
    def attached_to_token(cls, token, *args, **kwargs):
        instance = cls(*args, **kwargs)
        instance.attach_to_token(token)
        return instance

    def attach_to_token(self, token: FileContextToken):
        self.originates_from = token
        self.start_pos = token.start_pos
        self.line = token.line
        self.column = token.column
        self.end_line = token.end_line
        self.end_column = token.end_column
        self.end_pos = token.end_pos
