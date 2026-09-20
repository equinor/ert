from __future__ import annotations

from collections.abc import Iterator
from contextlib import contextmanager
from contextvars import ContextVar
from typing import TYPE_CHECKING, Any

from pydantic import BaseModel
from pydantic_core.core_schema import ValidationInfo

init_context_var = ContextVar("_init_context_var", default=None)

if TYPE_CHECKING:
    from ert.plugins import ErtRuntimePlugins


@contextmanager
def use_runtime_plugins(value: ErtRuntimePlugins) -> Iterator[None]:
    token = init_context_var.set(value)  # type: ignore
    try:
        yield
    finally:
        init_context_var.reset(token)


def get_runtime_plugins(info: ValidationInfo) -> ErtRuntimePlugins | None:
    """Return the active runtime plugins for a pydantic validator.

    When validating through FastAPI, the context is only available
    through init_context_var.
    """
    return info.context or init_context_var.get()


class BaseModelWithContextSupport(BaseModel, extra="forbid"):
    def __init__(__pydantic_self__, **data: Any) -> None:
        __pydantic_self__.__pydantic_validator__.validate_python(
            data,
            self_instance=__pydantic_self__,
            context=init_context_var.get(),
        )
