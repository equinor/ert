from __future__ import annotations

from collections.abc import Iterator
from contextlib import contextmanager
from contextvars import ContextVar
from typing import TYPE_CHECKING, Any

from pydantic import BaseModel

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

    def apply_context(
        self,
        site_runtime_plugins: ErtRuntimePlugins,
        user_runtime_plugins: ErtRuntimePlugins | None = None,
    ) -> BaseModelWithContextSupport:
        return self._apply_context_on_fn(
            self, site_runtime_plugins, user_runtime_plugins
        )

    def _apply_context_on_fn(
        self,
        obj: Any,
        site_runtime_plugins: ErtRuntimePlugins,
        user_runtime_plugins: ErtRuntimePlugins | None,
    ) -> Any:
        if callable(getattr(obj, "apply_context", None)) and obj is not self:
            return obj.apply_context(site_runtime_plugins, user_runtime_plugins)

        if isinstance(obj, BaseModel):
            new_data = {
                k: self._apply_context_on_fn(
                    v, site_runtime_plugins, user_runtime_plugins
                )
                for k, v in obj.__dict__.items()
            }
            return obj.__class__(**new_data)

        if isinstance(obj, dict):
            return {
                k: self._apply_context_on_fn(
                    v, site_runtime_plugins, user_runtime_plugins
                )
                for k, v in obj.items()
            }

        if isinstance(obj, list):
            return [
                self._apply_context_on_fn(i, site_runtime_plugins, user_runtime_plugins)
                for i in obj
            ]
        if isinstance(obj, tuple):
            return tuple(
                self._apply_context_on_fn(i, site_runtime_plugins, user_runtime_plugins)
                for i in obj
            )
        if isinstance(obj, set):
            return {
                self._apply_context_on_fn(i, site_runtime_plugins, user_runtime_plugins)
                for i in obj
            }

        return obj
