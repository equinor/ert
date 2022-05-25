from typing import Dict, List
from ._experiment_protocol import Experiment


class _Registry:
    def __init__(self) -> None:
        self._experiments: Dict[str, Experiment] = {}

    def add_experiment(self, experiment: Experiment) -> str:
        experiment_id = str(len(self._experiments.keys()))
        self._experiments[experiment_id] = experiment
        return experiment_id

    @property
    def all_experiments(self) -> List[Experiment]:
        return list(self._experiments.values())

    def get_experiment(self, experiment_id: str) -> Experiment:
        return self._experiments[experiment_id]
