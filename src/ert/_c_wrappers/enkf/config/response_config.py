import dataclasses
from abc import ABC
from typing import List

from sortedcontainers import SortedList


@dataclasses.dataclass
class ResponseConfig(ABC):
    name: str
    _observation_list: SortedList = SortedList()

    def getKey(self):
        return self.name

    def update_observation_keys(self, observations: List[str]):
        self._observation_list = SortedList(observations)

    def get_observation_keys(self) -> List[str]:
        return self._observation_list
