import queue
import uuid
from pathlib import Path
from unittest.mock import ANY, MagicMock, call

import pytest
from pydantic import ConfigDict

from ert.config import (
    ESSettings,
    ModelConfig,
    ObservationSettings,
    QueueConfig,
)
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
from ert.substitutions import Substitutions

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
def test_hook_call_order_ensemble_smoother(monkeypatch, use_tmpdir):
    """
    The goal of this test is to assert that the hook call order is the same
    across different models.
    """
    run_wfs_mock = MagicMock()
    monkeypatch.setattr(ensemble_smoother, "sample_prior", MagicMock())
    monkeypatch.setattr(ensemble_smoother, "smoother_update", MagicMock())
    monkeypatch.setattr(base_run_model.BaseRunModel, "run_workflows", run_wfs_mock)

    ens_mock = MagicMock()
    ens_mock.iteration = 0
    ens_mock.id = uuid.uuid1()
    storage_mock = MagicMock()
    storage_mock.create_ensemble.return_value = ens_mock

    class EnsembleSmootherWithMockSupport(EnsembleSmoother):
        model_config = ConfigDict(frozen=False, extra="allow")

    test_class = EnsembleSmootherWithMockSupport(
        target_ensemble="ens%d",
        experiment_name="exp",
        active_realizations=MagicMock(),
        minimum_required_realizations=MagicMock(),
        random_seed=0,
        storage_path="some_storage",
        queue_config=QueueConfig(),
        analysis_settings=ESSettings(),
        update_settings=ObservationSettings(),
        runpath_file=MagicMock(spec=Path),
        design_matrix=None,
        parameter_configuration=[],
        response_configuration=[],
        ert_templates=MagicMock(),
        user_config_file=MagicMock(spec=Path),
        env_vars=MagicMock(spec=dict),
        env_pr_fm_step=MagicMock(spec=dict),
        runpath_config=ModelConfig(),
        forward_model_steps=MagicMock(),
        substitutions=Substitutions(),
        hooked_workflows=MagicMock(spec=dict),
        log_path=Path(""),
        status_queue=queue.SimpleQueue(),
        observations=MagicMock(),
    )
    test_class.run_ensemble_evaluator = MagicMock(return_value=[0])
    test_class._storage = storage_mock
    test_class.run_experiment(MagicMock())

    assert run_wfs_mock.mock_calls == EXPECTED_CALL_ORDER


@pytest.mark.usefixtures("patch_base_run_model")
def test_hook_call_order_es_mda(monkeypatch, use_tmpdir):
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
    monkeypatch.setattr(multiple_data_assimilation, "smoother_update", MagicMock())
    monkeypatch.setattr(base_run_model.BaseRunModel, "run_workflows", run_wfs_mock)

    ens_mock = MagicMock()
    ens_mock.iteration = 0
    ens_mock.id = uuid.uuid1()
    storage_mock = MagicMock()
    storage_mock.create_ensemble.return_value = ens_mock

    class ESMDAWithMockSupport(MultipleDataAssimilation):
        model_config = ConfigDict(frozen=False, extra="allow")

    test_class = ESMDAWithMockSupport(
        target_ensemble="ens%d",
        restart_run=False,
        prior_ensemble_id="N/A",
        experiment_name="exp",
        weights="4,2,1",
        active_realizations=MagicMock(),
        minimum_required_realizations=MagicMock(),
        random_seed=0,
        storage_path="some_storage",
        queue_config=QueueConfig(),
        analysis_settings=ESSettings(),
        update_settings=ObservationSettings(),
        runpath_file=MagicMock(spec=Path),
        design_matrix=None,
        parameter_configuration=[],
        response_configuration=[],
        ert_templates=MagicMock(),
        user_config_file=MagicMock(spec=Path),
        env_vars=MagicMock(spec=dict),
        env_pr_fm_step=MagicMock(spec=dict),
        runpath_config=ModelConfig(),
        forward_model_steps=MagicMock(),
        substitutions=Substitutions(),
        hooked_workflows=MagicMock(spec=dict),
        log_path=Path(""),
        status_queue=queue.SimpleQueue(),
        observations=MagicMock(),
    )

    test_class._storage = storage_mock
    test_class.restart_run = False
    test_class.run_ensemble_evaluator = MagicMock(return_value=[0])
    test_class.run_experiment(MagicMock())

    assert run_wfs_mock.mock_calls == EXPECTED_CALL_ORDER
