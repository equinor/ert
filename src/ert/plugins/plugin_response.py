from __future__ import annotations

from typing import Generic, NamedTuple, TypeVar

T = TypeVar("T")


class PluginMetadata(NamedTuple):
    plugin_name: str
    function_name: str


class PluginResponse(Generic[T]):
    def __init__(self, data: T, plugin_metadata: PluginMetadata) -> None:
        self.data = data
        self.plugin_metadata = plugin_metadata
