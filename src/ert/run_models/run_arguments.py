from dataclasses import dataclass
from typing import List, Optional, Union


@dataclass
class SimulationArguments:
    random_seed: Optional[int]
    minimum_required_realizations: int
    ensemble_size: int
    stop_long_running: bool
    experiment_name: Optional[str]


@dataclass
class SingleTestRunArguments(SimulationArguments):
    current_ensemble: str = "prior"
    target_ensemble: Optional[str] = None
    ensemble_type: str = "Single test"

    def __post_init__(self) -> None:
        self.num_iterations = 1
        self.prev_successful_realizations = 0
        self.iter_num = 0
        self.start_iteration = 0
        self.active_realizations = [True]


@dataclass
class EnsembleExperimentRunArguments(SimulationArguments):
    active_realizations: List[bool]
    iter_num: int
    experiment_name: str
    start_iteration: int = 0
    current_ensemble: str = "prior"
    target_ensemble: Optional[str] = None
    ensemble_type: str = "Ensemble experiment"

    def __post_init__(self) -> None:
        self.num_iterations = 1
        self.prev_successful_realizations = 0


@dataclass
class EvaluateEnsembleRunArguments(SimulationArguments):
    active_realizations: List[bool]
    current_ensemble: str
    ensemble_type: str = "Evaluate ensemble"

    def __post_init__(self) -> None:
        self.target_ensemble = None
        self.iter_num = 0
        self.prev_successful_realizations = 0
        self.start_iteration = 0
        self.num_iterations = 1


@dataclass
class ESRunArguments(SimulationArguments):
    active_realizations: List[bool]
    current_ensemble: str
    target_ensemble: str
    ensemble_type: str = "ES"

    num_iterations: int = 1
    start_iteration: int = 0
    prev_successful_realizations: int = 0


# pylint: disable=R0902
@dataclass
class ESMDARunArguments(SimulationArguments):
    active_realizations: List[bool]
    target_ensemble: str
    weights: str
    restart_run: bool
    prior_ensemble: str

    ensemble_type: str = "ES_MDA"
    start_iteration: int = 0
    prev_successful_realizations: int = 0
    current_ensemble = None

    def __post_init__(self) -> None:
        self.num_iterations: int = len(self.weights)


@dataclass
class SIESRunArguments(SimulationArguments):
    active_realizations: List[bool]
    current_ensemble: str
    target_ensemble: str
    num_iterations: int
    num_retries_per_iter: int

    ensemble_type: str = "IES"
    start_iteration = 0
    iter_num = 0
    prev_successful_realizations = 0


RunArgumentsType = Union[
    SingleTestRunArguments,
    EnsembleExperimentRunArguments,
    EvaluateEnsembleRunArguments,
    ESRunArguments,
    ESMDARunArguments,
    SIESRunArguments,
]
