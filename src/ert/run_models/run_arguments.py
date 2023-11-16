from dataclasses import dataclass
from typing import List, Optional, Union


@dataclass
class SimulationArguments:
    random_seed: Optional[int]
    minimum_required_realizations: int
    ensemble_size: int
    stop_long_running: bool


@dataclass
class SingleTestRunArguments(SimulationArguments):
    current_case: str
    target_case: Optional[str] = None

    def __post_init__(self) -> None:
        self.num_iterations = 1
        self.prev_successful_realizations = 0
        self.iter_num = 0
        self.start_iteration = 0
        self.active_realizations = [True]


@dataclass
class EnsembleExperimentRunArguments(SimulationArguments):
    active_realizations: List[bool]
    current_case: str
    iter_num: int
    start_iteration: int = 0
    target_case: Optional[str] = None

    def __post_init__(self) -> None:
        self.num_iterations = 1
        self.prev_successful_realizations = 0


@dataclass
class ESRunArguments(SimulationArguments):
    active_realizations: List[bool]
    current_case: str
    target_case: str

    def __post_init__(self) -> None:
        self.analysis_module = "STD_ENKF"
        self.num_iterations = 1
        self.start_iteration = 0
        self.prev_successful_realizations = 0


# pylint: disable=R0902
@dataclass
class ESMDARunArguments(SimulationArguments):
    active_realizations: List[bool]
    target_case: str
    weights: str
    restart_run: bool
    prior_ensemble: str

    def __post_init__(self) -> None:
        self.start_iteration = 0
        self.prev_successful_realizations = 0
        self.current_case = None
        self.num_iterations = len(self.weights)


@dataclass
class SIESRunArguments(SimulationArguments):
    active_realizations: List[bool]
    current_case: str
    target_case: str
    num_iterations: int
    num_retries_per_iter: int

    def __post_init__(self) -> None:
        self.analysis_module = "IES_ENKF"
        self.start_iteration = 0
        self.iter_num = 0
        self.prev_successful_realizations = 0


RunArgumentsType = Union[
    SingleTestRunArguments,
    EnsembleExperimentRunArguments,
    ESRunArguments,
    ESMDARunArguments,
    SIESRunArguments,
]
