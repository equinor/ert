from __future__ import annotations

from dataclasses import dataclass
from typing import NamedTuple


class PluginMetadata(NamedTuple):
    plugin_name: str
    function_name: str


@dataclass
class PluginResponse[T]:
    data: T
    plugin_metadata: PluginMetadata
