from dataclasses import dataclass
from typing import Any, Generic, TypeVar

from decorator import decorator

T = TypeVar("T")


@dataclass
class PluginMetadata:
    plugin_name: str
    function_name: str


class PluginResponse(Generic[T]):
    def __init__(self, data: T, plugin_metadata: PluginMetadata) -> None:
        self.data = data
        self.plugin_metadata = plugin_metadata


@decorator
def plugin_response(
    func: Any, plugin_name: str = "", *args: Any, **kwargs: Any
) -> PluginResponse[T] | None:
    response = func(*args, **kwargs)
    return (
        PluginResponse(response, PluginMetadata(plugin_name, func.__name__))
        if response is not None
        else None
    )
