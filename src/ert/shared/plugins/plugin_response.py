from __future__ import annotations

from functools import wraps
from typing import Callable, Optional, TypeVar

from typing_extensions import ParamSpec

from ert.plugins.plugin_response import PluginMetadata, PluginResponse

T = TypeVar("T")
P = ParamSpec("P")


def plugin_response(
    plugin_name: str = "",
) -> Callable[[Callable[P, T]], Callable[P, Optional[PluginResponse[T]]]]:
    def outer(func: Callable[P, T]) -> Callable[P, Optional[PluginResponse[T]]]:
        @wraps(func)
        def inner(*args: P.args, **kwargs: P.kwargs) -> Optional[PluginResponse[T]]:
            response = func(*args, **kwargs)
            return (
                PluginResponse(response, PluginMetadata(plugin_name, func.__name__))
                if response is not None
                else None
            )

        return inner

    return outer
