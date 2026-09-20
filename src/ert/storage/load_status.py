from typing import NamedTuple, Self


class LoadResult(NamedTuple):
    successful: bool
    message: str

    @classmethod
    def success(cls, message: str = "") -> Self:
        return cls(True, message)

    @classmethod
    def failure(cls, message: str) -> Self:
        return cls(False, message)
