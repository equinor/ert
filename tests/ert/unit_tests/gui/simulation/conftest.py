from datetime import datetime
from uuid import uuid4

from ert.storage.local_ensemble import LocalEnsemble
from ert.storage.local_ensemble import _Index as _EnsembleIndex
from ert.storage.local_experiment import LocalExperiment
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
    def __init__(self, ensemble_name, experiment_id, storage_states, storage) -> None:
        self._index = _EnsembleIndex(
            id=uuid4(),
            experiment_id=experiment_id,
            ensemble_size=len(storage_states),
            iteration=0,
            name=ensemble_name,
            prior_ensemble_id=None,
            started_at=datetime.now(),
        )
        self._storage_state = storage_states
        self._storage = storage

    def get_ensemble_state(self):
        return self._storage_state


class MockExperiment(LocalExperiment):
    def __init__(self, experiment_name) -> None:
        self._index = _ExperimentIndex(id=uuid4(), name=experiment_name, ensembles=[])

    @property
    def relative_weights(self) -> str:
        return "4, 2, 1"


class MockStorage(LocalStorage):
    def __init__(self) -> None:
        self._ensembles = {}
        self._experiments = {}

    def _setup_mocked_run(
        self, ensemble_name, experiment_name, ensemble_states
    ) -> None:
        mock_experiment = MockExperiment(experiment_name)
        mock_ensemble2 = MockEnsemble(
            ensemble_name,
            experiment_id=mock_experiment.id,
            storage_states=ensemble_states,
            storage=self,
        )
        self._ensembles[mock_ensemble2.id] = mock_ensemble2
        self._experiments[mock_experiment.id] = mock_experiment
