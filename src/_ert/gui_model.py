import dataclasses
from typing import List, Literal, Mapping, Optional, Union
from uuid import UUID

import ert.config.analysis_module
from ert.config import ErtConfig
from ert.storage import open_storage


@dataclasses.dataclass
class ExperimentViewerModel:
    selected_experiment: Optional[UUID] = None
    selected_ensemble: Optional[UUID] = None
    selected_realization: int = 0


@dataclasses.dataclass
class RunAnalysisViewModel:
    target_ensemble: Optional[UUID] = None
    source_ensemble: Optional[UUID] = None


@dataclasses.dataclass
class LoadResultsManuallyModel:
    # "Load into ensemble"
    target_ensemble: Optional[UUID] = None
    realizations: List[int] = dataclasses.field(default_factory=list)
    iteration: int = 0


@dataclasses.dataclass
class ExperimentRunnerModel:
    """
    Union of options for all different experiments that can be run
    """

    experiment: UUID

    # Only for ensemble experiment (only one ensemble, thus not dynamic)
    new_single_ensemble_name: str
    new_dynamic_ensemble_name: str  # ies, ESMDA
    existing_ensemble_name: str

    active_realizations: List[bool]
    analysis_module: ert.config.analysis_module.AnalysisModule
    restart_run: bool
    restart_from_ensemble: UUID
    relative_weights: List[float] = dataclasses.field(
        default_factory=lambda _: [4, 2, 1]
    )


@dataclasses.dataclass
class ExperimentExplorerModel:
    experiment: UUID
    ensemble: Optional[UUID] = None
    realization: Optional[int] = None
    detail: Union[
        Mapping[
            Literal["experiment"],
            Literal["experiment", "observations", "parameters", "responses"],
        ],
        Mapping[
            Literal["ensemble"],
            Literal["ensemble", "state", "observations"],
        ],
        Mapping[
            Literal["realization"],
            Literal["realization"],
        ],
    ] = dataclasses.field(
        default_factory=lambda _: {
            "experiment": "experiment",
            "ensemble": "ensemble",
            "realization": "realization",
        }
    )


class GUIStorageModel:
    """
    Ideally all interactions between GUI / ert goes here
    (currently disregarding plotter, and snapshots of all the
    realizations while they run)
    """

    experiment_runner_model: ExperimentRunnerModel
    experiment_explorer_model: ExperimentExplorerModel
    load_results_manually_model: LoadResultsManuallyModel
    run_analysis_model: RunAnalysisViewModel

    def __init__(self, config: ErtConfig):
        with open_storage(config.ens_path) as storage:
            self._storage = storage

        self.experiment_runner_model = ExperimentRunnerModel()
        self.experiment_explorer_model = ExperimentExplorerModel()
        self.load_results_manually_model = LoadResultsManuallyModel()
        self.run_analysis_model = RunAnalysisViewModel()

    def create_experiment(self) -> UUID:
        pass

    def create_ensemble(self) -> UUID:
        pass

    def select_experiment(self, experiment: UUID) -> None:
        pass

    def select_ensemble(self, ensemble: UUID) -> None:
        pass

    @property
    def storage(self):
        pass
