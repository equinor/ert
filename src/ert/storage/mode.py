from __future__ import annotations

from collections.abc import Callable
from enum import StrEnum
from functools import wraps
from typing import Concatenate, Literal, ParamSpec, TypeVar

ModeLiteral = Literal["r", "w"]


class ModeError(ValueError):
    """
    Exception raised when an operation incompatible with the
    storage mode is attempted.
    """


class Mode(StrEnum):
    """Enumeration representing the access modes for storage interaction."""

    READ = "r"
    WRITE = "w"

    @property
    def can_write(self) -> bool:
        """Determine if the storage mode allows writing."""

        return self == self.WRITE


class BaseMode:
    """
    Base class for classes that require read/write access control to storage.

    This class provides a property to check if write operations are permitted
    and a method to assert write access before performing write operations.
    """

    def __init__(self, mode: Mode) -> None:
        """Initialize the base mode with the specified access mode.

        Parameters
        ----------
        mode : Mode
            The access mode for storage interaction.
        """

        self.__mode = mode

    @property
    def mode(self) -> Mode:
        return self.__mode

    @property
    def can_write(self) -> bool:
        return self.__mode.can_write

    def assert_can_write(self) -> None:
        """
        Assert that write operations are allowed under the current mode.

        Raises a ModeError if write operations are not allowed.
        """

        if not self.can_write:
            raise ModeError(
                "This operation requires write access, but we only have read access"
            )


P = ParamSpec("P")
T = TypeVar("T")
C = TypeVar("C", bound=BaseMode)
F = Callable[Concatenate[C, P], T]


def require_write(func: F[C, P, T]) -> F[C, P, T]:
    """
    Decorator to ensure a method can only be called in write mode.

    This decorator wraps a method to check if write operations are allowed
    before proceeding with the method call. If not, a ModeError is raised.

    Parameters
    ----------
    func : callable
        The method to wrap with write access enforcement.

    Returns
    -------
    inner : callable
        The wrapped method with write access enforcement.
    """

    @wraps(func)
    def inner(self_: C, /, *args: P.args, **kwargs: P.kwargs) -> T:
        self_.assert_can_write()
        return func(self_, *args, **kwargs)

    return inner
