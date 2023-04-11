from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from pathlib import Path

    from ert.storage import EnsembleAccessor, EnsembleReader


@dataclass
class ParameterConfig(ABC):
    name: str
    forward_init: bool

    @abstractmethod
    def load(self, run_path: Path, real_nr: int, ensemble: EnsembleAccessor):
        ...

    @abstractmethod
    def save(self, run_path: Path, real_nr: int, ensemble: EnsembleReader):
        ...

    def getKey(self):
        return self.name

    def getImplementationType(self):
        return self
