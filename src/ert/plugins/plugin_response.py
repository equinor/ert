from __future__ import annotations

from typing import Generic, TypeVar

T = TypeVar("T")


class PluginMetadata:
    def __init__(self, plugin_name: str, function_name: str) -> None:
        self.plugin_name = plugin_name
        self.function_name = function_name


class PluginResponse(Generic[T]):
    def __init__(self, data: T, plugin_metadata: PluginMetadata) -> None:
        self.data = data
        self.plugin_metadata = plugin_metadata
