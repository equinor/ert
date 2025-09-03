from __future__ import annotations

from json import JSONEncoder
from typing import Any, TypeVar, no_type_check

from .file_context_token import FileContextToken

# mypy: disable-error-code="attr-defined"


class ContextBoolEncoder(JSONEncoder):
    def default(self, o: Any) -> Any:
        return o.val if isinstance(o, ContextBool) else JSONEncoder.default(self, o)


class ContextBool:
    def __init__(self, val: bool, token: FileContextToken) -> None:
        self.val = val
        self.token = token

    def __bool__(self) -> bool:
        return bool(self.val)

    def __eq__(self, other: object) -> bool:
        return bool(self) == bool(other)

    def __repr__(self) -> str:
        if bool(self.val):
            return "True"
        return "False"

    @no_type_check
    def __deepcopy__(self, memo) -> ContextBool:
        new_instance = ContextBool(bool(self), self.token)
        memo[id(self)] = new_instance
        return new_instance


class ContextInt(int):
    def __new__(cls, val: int, token: FileContextToken) -> ContextInt:
        obj = super().__new__(cls, val)
        obj.token = token
        return obj

    @no_type_check
    def __deepcopy__(self, memo) -> ContextInt:
        new_instance = ContextInt(int(self), self.token)
        memo[id(self)] = new_instance
        return new_instance


class ContextFloat(float):
    def __new__(cls, val: float, token: FileContextToken) -> ContextFloat:
        obj = super().__new__(cls, val)
        obj.token = token
        return obj

    @no_type_check
    def __deepcopy__(self, memo) -> ContextFloat:
        new_instance = ContextFloat(float(self), self.token)
        memo[id(self)] = new_instance
        return new_instance


ContextString = FileContextToken


T = TypeVar("T")


class ContextList(list[T]):  # noqa: FURB189
    token: FileContextToken

    def __init__(self, token: FileContextToken) -> None:
        super().__init__()
        self.token = token

    @classmethod
    def with_values(
        cls, token: FileContextToken, values: list[ContextValue]
    ) -> ContextList[ContextValue]:
        the_list: ContextList[ContextValue] = ContextList(token)
        the_list += values
        return the_list


ContextValue = ContextString | ContextFloat | ContextInt | ContextBool
