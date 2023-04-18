from typing import Protocol

from .lark_parser_file_context_token import FileContextToken


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
        new_instance = BoolToken(bool(self), self.token, self.keyword_token)
        memo[id(self)] = new_instance
        return new_instance


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
        new_instance = StringToken(str(self), self.token, self.keyword_token)
        memo[id(self)] = new_instance
        return new_instance
