import dataclasses
from abc import ABC


@dataclasses.dataclass
class ResponseConfig(ABC):
    name: str

    def getKey(self):
        return self.name
