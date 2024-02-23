from __future__ import annotations

from enum import Enum
from functools import wraps
from typing import (
    TYPE_CHECKING,
    Callable,
    Literal,
)

if TYPE_CHECKING:
    from typing_extensions import Concatenate, ParamSpec, TypeVar

ModeLiteral = Literal["r", "w"]


class ModeError(ValueError):
    pass


class Mode(str, Enum):
    READ = "r"
    WRITE = "w"

    @property
    def can_write(self) -> bool:
        return self == self.WRITE


class BaseMode:
    """Base class inherited by classes that interact with storage.

    Classes that inherit ``BaseMode``, can assertain whether they are allowed
    to write to storage through the ``can_write`` property.

    Additionaly, through the ``@require_write`` (see :func:`~ert.storage.mode.require_write`) decorator,

    Parameters:
    -----------
    TODO
    """

    def __init__(self, mode: Mode) -> None:
        self.__mode = mode

    @property
    def mode(self) -> Mode:
        return self.__mode

    @property
    def can_write(self) -> bool:
        return self.__mode.can_write

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


def require_write(func: F[C, P, T]) -> F[C, P, T]:
    """Decorator that raises a ``ModeError`` (see :func:``~ert.storage.mode.BaseMode.assert_can_write`)
    if a wrapped method is called in read-only mode.
    """

    @wraps(func)
    def inner(self_: C, /, *args: P.args, **kwargs: P.kwargs) -> T:
        self_.assert_can_write()
        return func(self_, *args, **kwargs)

    return inner
