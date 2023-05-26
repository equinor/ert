from dataclasses import dataclass
from typing import Callable, List, Optional, Union


@dataclass
class DeprecationInfo:
    keyword: str
    message: Union[str, Callable[[List[str]], str]]
    check: Optional[Callable[[List[str]], bool]] = None

    def resolve_message(self, line: List[str]):
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
