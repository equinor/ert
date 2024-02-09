from __future__ import annotations

from enum import Enum
from functools import wraps
from typing import TYPE_CHECKING, Callable, Literal

if TYPE_CHECKING:
    from typing_extensions import Concatenate, ParamSpec, TypeVar

ModeLiteral = Literal["r", "w"]


class ModeError(ValueError):
    pass


class Mode(str, Enum):
    NONE = ""
    READ = "r"
    WRITE = "w"

    @property
    def can_read(self) -> bool:
        return self == self.READ

    @property
    def can_write(self) -> bool:
        return self == self.WRITE


class BaseMode:
    def __init__(self, mode: Mode) -> None:
        self.__mode = mode

    @property
    def mode(self) -> Mode:
        return self.__mode

    @property
    def can_read(self) -> bool:
        return self.__mode.can_write

    @property
    def can_write(self) -> bool:
        return self.__mode.can_write

    def reduce_mode(self, new_mode: Mode) -> None:
        """reduce mode

        Reduce mode ensuring that it is less permissive than the current mode

        Parameters
        ----------
        new_mode
            Mode to reduce to

        Raises
        ------
        ValueError
            If the new mode is more permissive. Eg, going from READ-enabled to WRITE-enabled.

        """
        if self.__mode == new_mode:
            return
        if new_mode == Mode.NONE or (
            new_mode == Mode.READ and self.__mode != Mode.NONE
        ):
            self.__mode = new_mode
        else:
            raise ValueError(
                f"Cannot increase mode from '{self.__mode}' to '{new_mode}'"
            )

    def assert_can_read(self) -> None:
        if not self.can_write:
            raise ModeError(
                "This operation requires write access, but we only have read access"
            )

    def assert_can_write(self) -> None:
        if not self.can_write:
            raise ModeError(
                "This operation requires write access, but we only have read access"
            )


if TYPE_CHECKING:
    P = ParamSpec("P")
    T = TypeVar("T")
    C = TypeVar("C", bound=BaseMode)
    F = Callable[Concatenate[C, P], T]


def require_read(func: F[C, P, T]) -> F[C, P, T]:
    @wraps(func)
    def inner(self_: C, /, *args: P.args, **kwargs: P.kwargs) -> T:
        self_.assert_can_read()
        return func(self_, *args, **kwargs)

    return inner


def require_write(func: F[C, P, T]) -> F[C, P, T]:
    @wraps(func)
    def inner(self_: C, /, *args: P.args, **kwargs: P.kwargs) -> T:
        self_.assert_can_write()
        return func(self_, *args, **kwargs)

    return inner
