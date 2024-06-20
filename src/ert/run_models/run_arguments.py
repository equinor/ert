from dataclasses import dataclass
from typing import List, Optional


@dataclass
class SimulationArguments:
    random_seed: Optional[int]
    minimum_required_realizations: int
    ensemble_size: int
    experiment_name: Optional[str]
    active_realizations: List[bool]


@dataclass
class SingleTestRunArguments(SimulationArguments):
    ensemble_name: str = "prior"
    ensemble_type: str = "Single test"


@dataclass
class EnsembleExperimentRunArguments(SimulationArguments):
    experiment_name: str
    ensemble_name: str = "prior"
    ensemble_type: str = "Ensemble experiment"


@dataclass
class EvaluateEnsembleRunArguments(SimulationArguments):
    ensemble_id: str
    ensemble_type: str = "Evaluate ensemble"


@dataclass
class ESRunArguments(SimulationArguments):
    target_ensemble: str
    ensemble_type: str = "ES"


# pylint: disable=R0902
@dataclass
class ESMDARunArguments(SimulationArguments):
    target_ensemble: str
    weights: str
    restart_run: bool
    prior_ensemble_id: str
    starting_iteration: int
    ensemble_type: str = "ES_MDA"
    current_ensemble = None


@dataclass
class SIESRunArguments(SimulationArguments):
    target_ensemble: str
    num_retries_per_iter: int
    number_of_iterations: int = 3
    ensemble_type: str = "IES"
