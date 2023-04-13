from dataclasses import dataclass, InitVar
from typing import List, Tuple, Union, cast, Optional, Protocol

from lark import Token


class FileContextToken(Token):
    """Represents a token, its location (line and column)
    within a file which is not necessarily the .ert config itself,
    but a file that is pointed to by the ert config."""

    filename: str

    # pylint: disable=signature-differs
    def __new__(cls, token: Token, filename: str) -> "FileContextToken":
        inst = super(FileContextToken, cls).__new__(
            cls,
            token.type,
            token.value,
            token.start_pos,
            token.line,
            token.column,
            token.end_line,
            token.end_column,
            token.end_pos,
        )

        inst_fct = cast(FileContextToken, inst)
        inst_fct.filename = filename
        return inst_fct

    def __repr__(self) -> str:
        return f"{self.value!r}"

    def __str__(self) -> str:
        return self.value

    @classmethod
    def join_tokens(
        cls, tokens: List["FileContextToken"], separator: str = " "
    ) -> "FileContextToken":
        first = tokens[0]
        min_start_pos = min(x.start_pos for x in tokens if x.start_pos is not None)
        max_end_pos = max(x.end_pos for x in tokens if x.end_pos is not None)
        min_line = min(x.line for x in tokens if x.line is not None)
        max_end_line = max(x.end_line for x in tokens if x.end_line is not None)
        min_column = min(
            x.column for x in tokens if x.line == min_line if x.column is not None
        )
        max_end_column = max(
            x.end_column
            for x in tokens
            if x.line == max_end_line
            if x.end_column is not None
        )
        return FileContextToken(
            Token(
                type=first.type,
                value=separator.join(tokens),
                start_pos=min_start_pos,
                line=min_line,
                column=min_column,
                end_line=max_end_line,
                end_column=max_end_column,
                end_pos=max_end_pos,
            ),
            filename=first.filename,
        )

    def replace_value(self, old: str, new: str, count: int = -1) -> "FileContextToken":
        replaced = self.value.replace(old, new, count)
        return FileContextToken(self.update(value=replaced), filename=self.filename)


# The type of the leaf nodes in the Tree after transformation is done
Instruction = List[
    List[Union[FileContextToken, List[Tuple[FileContextToken, FileContextToken]]]]
]

Defines = List[List[str]]


class BoolToken:
    def __init__(self, val: bool, token: str, keyword_token: FileContextToken):
        self.val = val
        self.token = token
        self.keyword_token = keyword_token

    def __str__(self):
        return bool(self.val)

    def __bool__(self):
        return bool(self.val)

    def __eq__(self, other):
        return bool(self) == bool(other)

    def __deepcopy__(self, memo):
        new_instance = BoolToken(bool(self), self.token)
        memo[id(self)] = new_instance
        return new_instance


class PrimitiveWithContext(Protocol):
    token: FileContextToken
    keyword_token: FileContextToken

    @classmethod
    def __subclasshook__(cls, subclass):
        allowed_subclasses = (IntToken, FloatToken, StringToken, BoolToken)
        allowed_attrs = ["token", "keyword_token"]

        if all(hasattr(subclass, x) for x in allowed_attrs) and isinstance(
            subclass, allowed_subclasses
        ):
            return True

        return NotImplemented


class IntToken(int):
    def __new__(
        cls, val: int, token: FileContextToken, keyword_token: FileContextToken
    ):
        obj = super().__new__(cls, val)
        obj.token = token
        obj.keyword_token = keyword_token
        return obj

    def __deepcopy__(self, memo):
        new_instance = IntToken(int(self), self.token, self.keyword_token)
        memo[id(self)] = new_instance
        return new_instance


class FloatToken(float):
    def __new__(
        cls, val: float, token: FileContextToken, keyword_token: FileContextToken
    ):
        obj = super().__new__(cls, val)
        obj.token = token
        obj.keyword_token = keyword_token
        return obj

    def __deepcopy__(self, memo):
        new_instance = FloatToken(float(self), self.token, self.keyword_token)
        memo[id(self)] = new_instance
        return new_instance


class StringToken(str):
    def __new__(
        cls, val: str, token: FileContextToken, keyword_token: FileContextToken
    ):
        obj = super().__new__(cls, val)
        obj.token = token
        obj.keyword_token = keyword_token
        return obj

    def __deepcopy__(self, memo):
        new_instance = StringToken(str(self), self.token, self.keyword_token)
        memo[id(self)] = new_instance
        return new_instance


class MaybeWithToken(Protocol):
    token: Optional["FileContextToken"]


class MaybeWithKeywordToken(Protocol):
    keyword_token: Optional["FileContextToken"]


TypedPrimitives = Union[FloatToken, BoolToken, StringToken, IntToken]
Primitives = Union[float, bool, str, int]
MaybeWithToken = Union[TypedPrimitives, Primitives]
MaybeWithKeywordToken = Union[TypedPrimitives, Primitives]


@dataclass
class ErrorInfo:
    filename: str
    message: str
    start_pos: Optional[int] = None
    line: Optional[int] = None
    column: Optional[int] = None
    end_line: Optional[int] = None
    end_column: Optional[int] = None
    end_pos: Optional[int] = None
    originates_from: InitVar[MaybeWithToken] = None
    originates_from_these: InitVar[List[MaybeWithToken]] = None
    originates_from_keyword: InitVar[MaybeWithKeywordToken] = None

    def __post_init__(
        self,
        originates_from: Optional[MaybeWithToken],
        originates_from_these: Optional[List[MaybeWithToken]],
        originates_from_keyword: Optional[MaybeWithKeywordToken],
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
