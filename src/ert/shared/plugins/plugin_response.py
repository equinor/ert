from __future__ import annotations

from functools import wraps
from typing import Callable, Generic, Optional, TypeVar

from typing_extensions import ParamSpec

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


class PluginMetadata:
    def __init__(self, plugin_name: str, function_name: str) -> None:
        self.plugin_name = plugin_name
        self.function_name = function_name


class PluginResponse(Generic[T]):
    def __init__(self, data: T, plugin_metadata: PluginMetadata) -> None:
        self.data = data
        self.plugin_metadata = plugin_metadata
