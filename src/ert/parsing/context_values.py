from typing import Dict, List, Optional, Tuple, TypeVar, Union

from .file_context_token import FileContextToken

# mypy: disable-error-code="attr-defined"


class ContextBool:
    def __init__(
        self, val: bool, token: FileContextToken, keyword_token: FileContextToken
    ):
        self.val = val
        self.token = token
        self.keyword_token = keyword_token

    def __bool__(self):
        return bool(self.val)

    def __eq__(self, other):
        return bool(self) == bool(other)

    def __deepcopy__(self, memo):
        new_instance = ContextBool(bool(self), self.token, self.keyword_token)
        memo[id(self)] = new_instance
        return new_instance


class ContextInt(int):
    def __new__(
        cls, val: int, token: FileContextToken, keyword_token: FileContextToken
    ):
        obj = super().__new__(cls, val)
        obj.token = token
        obj.keyword_token = keyword_token
        return obj

    def __deepcopy__(self, memo):
        new_instance = ContextInt(int(self), self.token, self.keyword_token)
        memo[id(self)] = new_instance
        return new_instance


class ContextFloat(float):
    def __new__(
        cls, val: float, token: FileContextToken, keyword_token: FileContextToken
    ):
        obj = super().__new__(cls, val)
        obj.token = token
        obj.keyword_token = keyword_token
        return obj

    def __deepcopy__(self, memo):
        new_instance = ContextFloat(float(self), self.token, self.keyword_token)
        memo[id(self)] = new_instance
        return new_instance


class ContextString(str):
    @classmethod
    def from_token(cls, token: FileContextToken):
        return cls(val=str(token), token=token, keyword_token=token)

    def __new__(
        cls, val: str, token: FileContextToken, keyword_token: FileContextToken
    ):
        obj = super().__new__(cls, val)
        obj.token = token
        obj.keyword_token = keyword_token
        return obj

    def __deepcopy__(self, memo):
        new_instance = ContextString(str(self), self.token, self.keyword_token)
        memo[id(self)] = new_instance
        return new_instance


T = TypeVar("T")


class ContextList(List[T]):
    keyword_token: FileContextToken
    defines: Optional["Defines"]

    def __init__(self, token: FileContextToken):
        super().__init__()
        self.keyword_token = token

    def attach_defines(self, defines: "Defines"):
        self.defines = defines


ContextValue = Union[ContextString, ContextFloat, ContextInt, ContextBool]

# The type of the leaf nodes in the Tree after transformation is done
Instruction = List[
    List[Union[FileContextToken, List[Tuple[FileContextToken, FileContextToken]]]]
]

Defines = List[List[str]]

Primitives = Union[float, bool, str, int]

# Old config parser gives primitives (which are without any context)
# while the new config parser gives primitives WITH context.
# Thus, we need a type to represent a union of these two as both
# parsers are in use.
MaybeWithContext = Union[ContextValue, Primitives, FileContextToken]


ConfigDict = Dict[
    str, Union[MaybeWithContext, List[MaybeWithContext], List[List[MaybeWithContext]]]
]
