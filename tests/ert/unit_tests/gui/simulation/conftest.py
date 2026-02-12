from datetime import datetime
from uuid import uuid4

from ert.storage.local_ensemble import LocalEnsemble
from ert.storage.local_ensemble import _Index as _EnsembleIndex
from ert.storage.local_experiment import ExperimentType, LocalExperiment
from ert.storage.local_experiment import _Index as _ExperimentIndex
from ert.storage.local_storage import LocalStorage
from ert.storage.realization_storage_state import RealizationStorageState

REALIZATION_FINISHED_SUCCESSFULLY = {
    RealizationStorageState.PARAMETERS_LOADED,
    RealizationStorageState.RESPONSES_LOADED,
}
REALIZATION_UNDEFINED = {RealizationStorageState.UNDEFINED}
REALIZATION_ONLY_PARAMETERS = {RealizationStorageState.PARAMETERS_LOADED}
REALIZATION_FAILED_DURING_EVALUATION = {
    RealizationStorageState.PARAMETERS_LOADED,
    RealizationStorageState.FAILURE_IN_CURRENT,
}


class MockEnsemble(LocalEnsemble):
    def __init__(
        self, ensemble_name, experiment_id, storage_states, storage, iteration=0
    ) -> None:
        self._index = _EnsembleIndex(
            id=uuid4(),
            experiment_id=experiment_id,
            ensemble_size=len(storage_states),
            iteration=iteration,
            name=ensemble_name,
            prior_ensemble_id=None,
            started_at=datetime.now(),
        )
        self._storage_state = storage_states
        self._storage = storage

    def get_ensemble_state(self):
        return self._storage_state


class MockExperiment(LocalExperiment):
    def __init__(self, experiment_name, experiment_type) -> None:
        self._index = _ExperimentIndex(id=uuid4(), name=experiment_name, ensembles=[])
        self._type = experiment_type

    @property
    def relative_weights(self) -> str:
        if self._type in {ExperimentType.ES_MDA, ExperimentType.UNDEFINED}:
            return "4, 2, 1"
        return ""

    @property
    def experiment_type(self) -> ExperimentType:
        return self._type


class MockStorage(LocalStorage):
    def __init__(self) -> None:
        self._ensembles = {}
        self._experiments = {}

    def _setup_mocked_run(
        self,
        ensemble_name,
        experiment_name,
        ensemble_states,
        *,
        iteration=0,
        experiment_type=ExperimentType.UNDEFINED,
    ) -> None:
        mock_experiment = MockExperiment(experiment_name, experiment_type)
        mock_ensemble2 = MockEnsemble(
            ensemble_name,
            experiment_id=mock_experiment.id,
            storage_states=ensemble_states,
            storage=self,
            iteration=iteration,
        )
        self._ensembles[mock_ensemble2.id] = mock_ensemble2
        self._experiments[mock_experiment.id] = mock_experiment
