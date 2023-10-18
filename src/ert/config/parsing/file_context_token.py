from typing import List, cast

from lark import Token


class FileContextToken(Token):
    """Represents a token, its location (line and column)
    within a file which is not necessarily the .ert config itself,
    but a file that is pointed to by the ert config."""

    filename: str

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

    def __eq__(self, other: object) -> bool:
        if isinstance(other, str):
            return self.value == other
        if isinstance(other, Token):
            return self.value == other.value
        else:
            return False

    def __hash__(self) -> int:  # type: ignore
        return hash(self.value)

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
        if old in self.value:
            replaced = self.value.replace(old, new, count)
            return FileContextToken(self.update(value=replaced), filename=self.filename)
        return self
