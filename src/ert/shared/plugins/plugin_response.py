from __future__ import annotations

from collections.abc import Callable
from functools import wraps
from typing import ParamSpec, TypeVar

from ert.plugins.plugin_response import PluginMetadata, PluginResponse

T = TypeVar("T")
P = ParamSpec("P")


def plugin_response(
    plugin_name: str = "",
) -> Callable[[Callable[P, T]], Callable[P, PluginResponse[T] | None]]:
    def outer(func: Callable[P, T]) -> Callable[P, PluginResponse[T] | None]:
        @wraps(func)
        def inner(*args: P.args, **kwargs: P.kwargs) -> PluginResponse[T] | None:
            response = func(*args, **kwargs)
            return (
                PluginResponse(response, PluginMetadata(plugin_name, func.__name__))
                if response is not None
                else None
            )

        return inner

    return outer
