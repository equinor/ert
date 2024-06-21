from dataclasses import dataclass
from typing import List, Optional


@dataclass
class SimulationArguments:
    random_seed: Optional[int]
    minimum_required_realizations: int
    ensemble_size: int
    active_realizations: List[bool]


@dataclass
class EnsembleExperimentRunArguments(SimulationArguments):
    experiment_name: str
    ensemble_name: str = "prior"
    ensemble_type: str = "Ensemble experiment"


@dataclass
class SingleTestRunArguments(EnsembleExperimentRunArguments):
    experiment_name: str
    ensemble_name: str = "prior"
    ensemble_type: str = "Single test"


@dataclass
class EvaluateEnsembleRunArguments(SimulationArguments):
    ensemble_id: str
    ensemble_type: str = "Evaluate ensemble"


@dataclass
class ESRunArguments(SimulationArguments):
    target_ensemble: str
    experiment_name: str
    ensemble_type: str = "ES"


# pylint: disable=R0902
@dataclass
class ESMDARunArguments(SimulationArguments):
    target_ensemble: str
    weights: str
    restart_run: bool
    prior_ensemble_id: str
    starting_iteration: int
    experiment_name: Optional[str]
    ensemble_type: str = "ES_MDA"


@dataclass
class SIESRunArguments(SimulationArguments):
    target_ensemble: str
    num_retries_per_iter: int
    experiment_name: str
    number_of_iterations: int = 3
    ensemble_type: str = "IES"
