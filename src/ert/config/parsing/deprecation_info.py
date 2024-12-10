from collections.abc import Callable
from dataclasses import dataclass

from .context_values import ContextValue


@dataclass
class DeprecationInfo:
    keyword: str
    message: str | Callable[[list[str]], str]
    check: Callable[[list[ContextValue]], bool] | None = None

    def resolve_message(self, line: list[str]) -> str:
        if callable(self.message):
            return self.message(line)

        return self.message

    @staticmethod
    def is_angle_bracketed(keyword: str) -> bool:
        return (
            keyword.count("<") + keyword.count(">") == 2
            and keyword.startswith("<")
            and keyword.endswith(">")
        )
