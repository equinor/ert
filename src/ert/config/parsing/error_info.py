from collections.abc import Sequence
from dataclasses import dataclass
from typing import Self

from .file_context_token import FileContextToken
from .types import MaybeWithContext


@dataclass
class ErrorInfo:
    message: str
    filename: str | None = None
    start_pos: int | None = None
    line: int | None = None
    column: int | None = None
    end_line: int | None = None
    end_column: int | None = None
    end_pos: int | None = None

    @classmethod
    def _take(cls, context: MaybeWithContext, attr: str) -> FileContextToken | None:
        if isinstance(context, FileContextToken):
            return context
        elif hasattr(context, attr):
            return getattr(context, attr)

        return None

    def set_context(self, context: MaybeWithContext) -> Self:
        self._attach_to_context(self._take(context, "token"))
        return self

    def set_context_keyword(self, context: MaybeWithContext) -> Self:
        self._attach_to_context(self._take(context, "keyword_token"))
        return self

    def set_context_list(self, context_list: Sequence[MaybeWithContext]) -> Self:
        parsed_context_list = []
        for context in context_list:
            the_context = self._take(context, attr="token")
            if the_context is not None:
                parsed_context_list.append(the_context)

        if len(parsed_context_list) > 0:
            context = FileContextToken.join_tokens(parsed_context_list)
            self._attach_to_context(context)

        return self

    def __gt__(self, other: "ErrorInfo") -> bool:
        for attr in [
            "filename",
            "line",
            "column",
            "start_pos",
            "end_pos",
            "end_line",
            "end_pos",
        ]:
            if getattr(self, attr) is not None:
                if getattr(other, attr) is None:
                    return True
                return getattr(self, attr) > getattr(other, attr)
        return self.message > other.message

    def location(self) -> str:
        msg = ""
        if self.filename is not None:
            msg += f"{self.filename}: "
        if self.line is not None:
            msg += f"Line {self.line} "
        if self.column is not None and self.end_column is None:
            msg += f"(Column {self.column}): "
        if self.column is not None and self.end_column is not None:
            msg += f"(Column {self.column}-{self.end_column}): "
        return msg

    def __str__(self) -> str:
        return self.location() + self.message

    def _attach_to_context(self, token: FileContextToken | None) -> None:
        if token is not None:
            self.filename = token.filename
            self.start_pos = token.start_pos
            self.line = token.line
            self.column = token.column
            self.end_line = token.end_line
            self.end_column = token.end_column
            self.end_pos = token.end_pos


@dataclass()
class WarningInfo(ErrorInfo):
    is_deprecation: bool = False
