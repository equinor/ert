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
    PostExperimentFixtures,
    PostSimulationFixtures,
    PostUpdateFixtures,
    PreExperimentFixtures,
    PreFirstUpdateFixtures,
    PreSimulationFixtures,
    PreUpdateFixtures,
    QueueConfig,
    fixtures_per_hook,
)
from ert.run_models import (
    EnsembleExperiment,
    EnsembleInformationFilter,
    EnsembleSmoother,
    MultipleDataAssimilation,
    RunModel,
    initial_ensemble_run_model,
    run_model,
)

EXPECTED_CALL_ORDER_WITH_ONE_UPDATE = [
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

EXPECTED_CALL_ORDER_WITHOUT_UPDATES = [
    call(
        fixtures=cls(**dict.fromkeys(fixtures_per_hook[cls.hook], ANY), hook=cls.hook),
    )
    for cls in [
        PreExperimentFixtures,
        PreSimulationFixtures,
        PostSimulationFixtures,
        PostExperimentFixtures,
    ]
]


@pytest.fixture
def patch_run_model(monkeypatch):
    monkeypatch.setattr(run_model, "create_run_path", MagicMock())
    monkeypatch.setattr(RunModel, "validate_successful_realizations_count", MagicMock())
    monkeypatch.setattr(RunModel, "set_env_key", MagicMock())


@pytest.mark.parametrize(
    ("cls", "extra_args", "expected_call_order"),
    [
        (
            MultipleDataAssimilation,
            {
                "weights": "1",
                "target_ensemble": "ens%d",
                "restart_run": False,
                "prior_ensemble_id": "N/A",
            },
            EXPECTED_CALL_ORDER_WITH_ONE_UPDATE,
        ),
        (
            EnsembleSmoother,
            {"target_ensemble": "ens%d"},
            EXPECTED_CALL_ORDER_WITH_ONE_UPDATE,
        ),
        (
            EnsembleExperiment,
            {"target_ensemble": "ens"},
            EXPECTED_CALL_ORDER_WITHOUT_UPDATES,
        ),
        (
            EnsembleInformationFilter,
            {"target_ensemble": "ens%d"},
            EXPECTED_CALL_ORDER_WITH_ONE_UPDATE,
        ),
    ],
)
@pytest.mark.usefixtures("patch_run_model")
def test_hook_call_order(monkeypatch, use_tmpdir, cls, extra_args, expected_call_order):
    """
    The goal of this test is to assert that the hook call order is the same
    across different models.
    """
    run_wfs_mock = MagicMock()
    monkeypatch.setattr(initial_ensemble_run_model, "sample_prior", MagicMock())
    monkeypatch.setattr(cls, "update_ensemble_parameters", MagicMock(), raising=False)
    monkeypatch.setattr(run_model.RunModel, "run_workflows", run_wfs_mock)

    ens_mock = MagicMock()
    ens_mock.iteration = 0
    ens_mock.id = uuid.uuid1()
    storage_mock = MagicMock()
    storage_mock.create_ensemble.return_value = ens_mock

    class ModelWithMockSupport(cls):
        model_config = ConfigDict(frozen=False, extra="allow")

    test_class = ModelWithMockSupport(
        experiment_name="exp",
        active_realizations=MagicMock(),
        minimum_required_realizations=MagicMock(),
        random_seed=0,
        **extra_args,
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
        substitutions={},
        hooked_workflows=MagicMock(spec=dict),
        log_path=Path(""),
        status_queue=queue.SimpleQueue(),
        observations=None,
    )

    test_class.run_ensemble_evaluator = MagicMock(return_value=[0])
    test_class._storage = storage_mock
    test_class.run_experiment(MagicMock())

    assert run_wfs_mock.mock_calls == expected_call_order
