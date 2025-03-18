import uuid
from unittest.mock import ANY, MagicMock, call

import pytest

from ert.plugins import (
    PostExperimentFixtures,
    PostSimulationFixtures,
    PostUpdateFixtures,
    PreExperimentFixtures,
    PreFirstUpdateFixtures,
    PreSimulationFixtures,
    PreUpdateFixtures,
    fixtures_per_hook,
)
from ert.run_models import (
    BaseRunModel,
    EnsembleSmoother,
    MultipleDataAssimilation,
    base_run_model,
    ensemble_smoother,
    multiple_data_assimilation,
)

EXPECTED_CALL_ORDER = [
    call(
        fixtures=cls(**dict.fromkeys(fixtures_per_hook[cls.hook], ANY), hook=cls.hook),
    )
    for cls in [
        PreExperimentFixtures,
        PreSimulationFixtures,
        PostSimulationFixtures,
        PreFirstUpdateFixtures,
        PreUpdateFixtures,
        PostUpdateFixtures,
        PreSimulationFixtures,
        PostSimulationFixtures,
        PostExperimentFixtures,
    ]
]


@pytest.fixture
def patch_base_run_model(monkeypatch):
    monkeypatch.setattr(base_run_model, "create_run_path", MagicMock())
    monkeypatch.setattr(
        BaseRunModel, "validate_successful_realizations_count", MagicMock()
    )
    monkeypatch.setattr(BaseRunModel, "set_env_key", MagicMock())


@pytest.mark.usefixtures("patch_base_run_model")
def test_hook_call_order_ensemble_smoother(monkeypatch):
    """
    The goal of this test is to assert that the hook call order is the same
    across different models.
    """
    run_wfs_mock = MagicMock()
    monkeypatch.setattr(ensemble_smoother, "sample_prior", MagicMock())
    monkeypatch.setattr(base_run_model, "smoother_update", MagicMock())
    monkeypatch.setattr(base_run_model.BaseRunModel, "run_workflows", run_wfs_mock)

    ens_mock = MagicMock()
    ens_mock.iteration = 0
    ens_mock.id = uuid.uuid1()
    storage_mock = MagicMock()
    storage_mock.create_ensemble.return_value = ens_mock

    test_class = EnsembleSmoother(
        target_ensemble=MagicMock(),
        experiment_name=MagicMock(),
        active_realizations=MagicMock(),
        minimum_required_realizations=MagicMock(),
        random_seed=0,
        config=MagicMock(),
        storage=MagicMock(),
        queue_config=MagicMock(),
        es_settings=MagicMock(),
        update_settings=MagicMock(),
        status_queue=MagicMock(),
    )
    test_class.run_ensemble_evaluator = MagicMock(return_value=[0])
    test_class._storage = storage_mock
    test_class._design_matrix = None
    test_class.run_experiment(MagicMock())

    assert run_wfs_mock.mock_calls == EXPECTED_CALL_ORDER


@pytest.mark.usefixtures("patch_base_run_model")
def test_hook_call_order_es_mda(monkeypatch):
    """
    The goal of this test is to assert that the hook call order is the same
    across different models.
    """

    run_wfs_mock = MagicMock()
    monkeypatch.setattr(multiple_data_assimilation, "sample_prior", MagicMock())
    monkeypatch.setattr(
        multiple_data_assimilation.MultipleDataAssimilation,
        "parse_weights",
        MagicMock(return_value=[1]),
    )
    monkeypatch.setattr(base_run_model, "smoother_update", MagicMock())
    monkeypatch.setattr(base_run_model.BaseRunModel, "run_workflows", run_wfs_mock)

    ens_mock = MagicMock()
    ens_mock.iteration = 0
    ens_mock.id = uuid.uuid1()
    storage_mock = MagicMock()
    storage_mock.create_ensemble.return_value = ens_mock
    test_class = MultipleDataAssimilation(
        target_ensemble=MagicMock(),
        experiment_name=MagicMock(),
        restart_run=MagicMock(),
        prior_ensemble_id=MagicMock(),
        active_realizations=MagicMock(),
        minimum_required_realizations=MagicMock(),
        random_seed=0,
        weights=MagicMock(),
        config=MagicMock(),
        storage=MagicMock(),
        queue_config=MagicMock(),
        es_settings=MagicMock(),
        update_settings=MagicMock(),
        status_queue=MagicMock(),
    )
    test_class._storage = storage_mock
    test_class.restart_run = False
    test_class.run_ensemble_evaluator = MagicMock(return_value=[0])
    test_class._design_matrix = None
    test_class.run_experiment(MagicMock())

    assert run_wfs_mock.mock_calls == EXPECTED_CALL_ORDER
