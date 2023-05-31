from dataclasses import dataclass
from typing import List, Optional, Union

from typing_extensions import Self

from .context_values import MaybeWithContext
from .file_context_token import FileContextToken


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
    originates_from: Optional[MaybeWithContext] = None

    @classmethod
    def _take(cls, context: MaybeWithContext, attr: str) -> Optional[FileContextToken]:
        if isinstance(context, FileContextToken):
            return context
        elif hasattr(context, attr):
            return getattr(context, attr)

        return None

    def set_context(self, context: MaybeWithContext) -> Self:
        self._attach_to_context(self._take(context, "token"))
        return self

    def set_context_keyword(
        self, context: Union[MaybeWithContext, List[MaybeWithContext]]
    ) -> Self:
        if isinstance(context, List):
            # If it is a list, each item is an argument
            # pertaining to the same keyword token written in an ert config.
            # Thus, it is ok to take the first item only.
            first_item = context[0]
            keyword_token = self._take(first_item, "keyword_token")

            self._attach_to_context(keyword_token)
        else:
            self._attach_to_context(self._take(context, "keyword_token"))

        return self

    def set_context_list(self, context_list: List[MaybeWithContext]) -> Self:
        parsed_context_list = []
        for context in context_list:
            the_context = self._take(context, attr="token")
            if the_context is not None:
                parsed_context_list.append(the_context)

        if len(parsed_context_list) > 0:
            context = FileContextToken.join_tokens(parsed_context_list)
            self._attach_to_context(context)

        return self

    @property
    def message_with_location(self):
        return (
            self.message
            + f" at line {self.line}, column {self.column}-{self.end_column}"
        )

    @classmethod
    def attached_to_context(cls, token, *args, **kwargs):
        instance = cls(*args, **kwargs)
        instance._attach_to_context(token)
        return instance

    def _attach_to_context(self, token: Optional[FileContextToken]):
        if token is not None:
            self.originates_from = token
            self.start_pos = token.start_pos
            self.line = token.line
            self.column = token.column
            self.end_line = token.end_line
            self.end_column = token.end_column
            self.end_pos = token.end_pos
