from typing import List


class SummaryConfig:
    def __init__(self, key):
        self.name = key
        self._observation_list: List[str] = []

    def update_observation_keys(self, observations: List[str]):
        self._observation_list = observations
        self._observation_list.sort()

    def get_observation_keys(self) -> List[str]:
        return self._observation_list

    def getKey(self):
        return self.name

    def __repr__(self):
        return (
            f"SummaryConfig(key={self.name}, "
            f"observation_keys={self._observation_list})"
        )
