from abc import ABC
from typing import Any, List

from .ert_script import ErtScript


class CancelPluginException(Exception):
    """Raised when a plugin is cancelled."""


class ErtPlugin(ErtScript, ABC):

    @staticmethod
    def getArguments(parent: Any = None) -> List[Any]:
        return []

    def getName(self) -> str:
        return str(self.__class__)

    @staticmethod
    def getDescription() -> str:
        return "No description provided!"
