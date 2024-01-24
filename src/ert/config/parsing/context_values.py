from json import JSONEncoder
from typing import Any, List, TypeVar, Union, no_type_check

from .file_context_token import FileContextToken

# mypy: disable-error-code="attr-defined"


class ContextBoolEncoder(JSONEncoder):
    def default(self, o: Any) -> Any:
        return o.val if isinstance(o, ContextBool) else JSONEncoder.default(self, o)


class ContextBool:
    def __init__(
        self, val: bool, token: FileContextToken, keyword_token: FileContextToken
    ) -> None:
        self.val = val
        self.token = token
        self.keyword_token = keyword_token

    def __bool__(self) -> bool:
        return bool(self.val)

    def __eq__(self, other: object) -> bool:
        return bool(self) == bool(other)

    @no_type_check
    def __deepcopy__(self, memo):
        new_instance = ContextBool(bool(self), self.token, self.keyword_token)
        memo[id(self)] = new_instance
        return new_instance


class ContextInt(int):
    def __new__(
        cls, val: int, token: FileContextToken, keyword_token: FileContextToken
    ) -> "ContextInt":
        obj = super().__new__(cls, val)
        obj.token = token
        obj.keyword_token = keyword_token
        return obj

    @no_type_check
    def __deepcopy__(self, memo):
        new_instance = ContextInt(int(self), self.token, self.keyword_token)
        memo[id(self)] = new_instance
        return new_instance


class ContextFloat(float):
    def __new__(
        cls, val: float, token: FileContextToken, keyword_token: FileContextToken
    ) -> "ContextFloat":
        obj = super().__new__(cls, val)
        obj.token = token
        obj.keyword_token = keyword_token
        return obj

    @no_type_check
    def __deepcopy__(self, memo):
        new_instance = ContextFloat(float(self), self.token, self.keyword_token)
        memo[id(self)] = new_instance
        return new_instance


class ContextString(str):
    @classmethod
    def from_token(cls, token: FileContextToken) -> "ContextString":
        return cls(val=str(token), token=token, keyword_token=token)

    def __new__(
        cls, val: str, token: FileContextToken, keyword_token: FileContextToken
    ) -> "ContextString":
        obj = super().__new__(cls, val)
        obj.token = token
        obj.keyword_token = keyword_token
        return obj

    @no_type_check
    def __deepcopy__(self, memo):
        new_instance = ContextString(str(self), self.token, self.keyword_token)
        memo[id(self)] = new_instance
        return new_instance


T = TypeVar("T")


class ContextList(List[T]):
    keyword_token: FileContextToken

    def __init__(self, token: FileContextToken) -> None:
        super().__init__()
        self.keyword_token = token

    @classmethod
    def with_values(
        cls, token: FileContextToken, values: List["ContextValue"]
    ) -> "ContextList[ContextValue]":
        the_list: "ContextList[ContextValue]" = ContextList(token)
        the_list += values
        return the_list


ContextValue = Union[ContextString, ContextFloat, ContextInt, ContextBool]
