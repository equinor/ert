from dataclasses import InitVar, dataclass
from typing import List, Optional

from .lark_parser_file_context_token import FileContextToken
from .lark_parser_types import MaybeWithContext


@dataclass
# pylint: disable=too-many-instance-attributes
class ErrorInfo:
    filename: str
    message: str
    start_pos: Optional[int] = None
    line: Optional[int] = None
    column: Optional[int] = None
    end_line: Optional[int] = None
    end_column: Optional[int] = None
    end_pos: Optional[int] = None
    originates_from: InitVar[MaybeWithContext] = None
    originates_from_these: InitVar[List[MaybeWithContext]] = None
    originates_from_keyword: InitVar[MaybeWithContext] = None

    def __post_init__(
        self,
        originates_from: Optional[MaybeWithContext],
        originates_from_these: Optional[List[MaybeWithContext]],
        originates_from_keyword: Optional[MaybeWithContext],
    ):
        token = None
        if originates_from_keyword is not None and hasattr(
            originates_from_keyword, "keyword_token"
        ):
            token = originates_from_keyword.keyword_token
        elif originates_from is not None and hasattr(originates_from, "token"):
            token = originates_from.token
        elif originates_from_these is not None:
            tokens = [x.token for x in originates_from_these if hasattr(x, "token")]
            # Merge the token positions etc

            if len(tokens) > 0:
                token = FileContextToken.join_tokens(tokens)

        if token is not None:
            self.start_pos = token.start_pos
            self.line = token.line
            self.column = token.column
            self.end_line = token.end_line
            self.end_column = token.end_column
            self.end_pos = token.end_pos
        pass
