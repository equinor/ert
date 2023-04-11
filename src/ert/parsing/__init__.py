from typing import List, Optional, Tuple, Union

from lark import Token


class FileContextToken(Token):
    """Represents a token, its location (line and column)
    within a file which is not necessarily the .ert config itself,
    but a file that is pointed to by the ert config."""

    filename: str

    # pylint: disable=signature-differs
    def __new__(cls, token, filename):
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
        inst.filename = filename
        return inst

    def __repr__(self):
        return f"{self.value!r}"

    def __str__(self):
        return self.value

    @classmethod
    def join_tokens(
        cls, tokens: List["FileContextToken"], separator=" "
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

    def replace(self, old: str, new: str, count=-1):
        replaced = self.value.replace(old, new, count)
        return FileContextToken(self.update(value=replaced), filename=self.filename)


# The type of the leaf nodes in the Tree after transformation is done
Instruction = List[
    List[Union[FileContextToken, List[Tuple[FileContextToken, FileContextToken]]]]
]

Defines = List[List[str]]


class ConfigWarning(UserWarning):
    pass


class ConfigValidationError(ValueError):
    def __init__(self, errors: str, config_file: Optional[str] = None) -> None:
        self.config_file = config_file
        self.errors = errors
        super().__init__(
            (
                f"Parsing config file `{self.config_file}` "
                f"resulted in the errors: {self.errors}"
            )
            if self.config_file
            else f"{self.errors}"
        )
