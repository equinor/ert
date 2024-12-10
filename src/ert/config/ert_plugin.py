from __future__ import annotations

from abc import ABC
from typing import Any

from .ert_script import ErtScript


class CancelPluginException(Exception):
    """Raised when a plugin is cancelled."""


class ErtPlugin(ErtScript, ABC):
    def getArguments(self, args: list[Any]) -> list[Any]:
        return []

    def getName(self) -> str:
        return str(self.__class__)

    def getDescription(self) -> str:
        return "No description provided!"
