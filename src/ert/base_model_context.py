from __future__ import annotations

from collections.abc import Callable, Iterator
from contextlib import contextmanager
from contextvars import ContextVar
from functools import wraps
from typing import TYPE_CHECKING, Any, TypeVar

from pydantic import BaseModel, model_validator
from pydantic_core.core_schema import ValidationInfo

init_context_var = ContextVar("_init_context_var", default=None)

if TYPE_CHECKING:
    from ert.plugins import ErtRuntimePlugins


@contextmanager
def init_context(value: ErtRuntimePlugins) -> Iterator[None]:
    token = init_context_var.set(value)  # type: ignore
    try:
        yield
    finally:
        init_context_var.reset(token)


class BaseModelWithContextSupport(BaseModel):
    def __init__(__pydantic_self__, **data: Any) -> None:
        __pydantic_self__.__pydantic_validator__.validate_python(
            data,
            self_instance=__pydantic_self__,
            context=init_context_var.get(),
        )

    @classmethod
    def with_plugins(
        cls, plugins: list[ErtRuntimePlugins], **data: Any
    ) -> BaseModelWithContextSupport:
        current = cls(**data)
        for runtime_plugins in plugins:
            with init_context(runtime_plugins):
                current = cls.model_copy()

        return current


T = TypeVar("T", bound=BaseModelWithContextSupport)


def when_plugins(*, mode: str = "after"):
    """
    @model_validator that only executes
    if ValidationInfo.context is ErtRunTimePlugins.
    """

    def decorator(fn: Callable[[type[T], T, ValidationInfo], T]) -> Callable:
        @model_validator(mode=mode)
        @wraps(fn)
        def wrapper(cls, values: T, info: ValidationInfo) -> T:
            if isinstance(info.context, ErtRuntimePlugins):
                return fn(cls, values, info)
            return values

        return wrapper

    return decorator
