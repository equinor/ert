import uuid
from unittest.mock import ANY, MagicMock, call

import pytest

from ert.config import HookRuntime
from ert.run_models import (
    BaseRunModel,
    EnsembleSmoother,
    MultipleDataAssimilation,
    base_run_model,
    ensemble_smoother,
    multiple_data_assimilation,
)

EXPECTED_CALL_ORDER = [
    call(HookRuntime.PRE_EXPERIMENT, fixtures={"random_seed": ANY}),
    call(
        HookRuntime.PRE_SIMULATION,
        fixtures={
            "storage": ANY,
            "ensemble": ANY,
            "reports_dir": ANY,
            "random_seed": ANY,
            "run_paths": ANY,
        },
    ),
    call(
        HookRuntime.POST_SIMULATION,
        fixtures={
            "storage": ANY,
            "ensemble": ANY,
            "reports_dir": ANY,
            "random_seed": ANY,
            "run_paths": ANY,
        },
    ),
    call(
        HookRuntime.PRE_FIRST_UPDATE,
        fixtures={
            "storage": ANY,
            "ensemble": ANY,
            "reports_dir": ANY,
            "random_seed": ANY,
            "es_settings": ANY,
            "observation_settings": ANY,
            "run_paths": ANY,
        },
    ),
    call(
        HookRuntime.PRE_UPDATE,
        fixtures={
            "storage": ANY,
            "ensemble": ANY,
            "reports_dir": ANY,
            "random_seed": ANY,
            "es_settings": ANY,
            "observation_settings": ANY,
            "run_paths": ANY,
        },
    ),
    call(
        HookRuntime.POST_UPDATE,
        fixtures={
            "storage": ANY,
            "ensemble": ANY,
            "reports_dir": ANY,
            "random_seed": ANY,
            "es_settings": ANY,
            "observation_settings": ANY,
            "run_paths": ANY,
        },
    ),
    call(
        HookRuntime.PRE_SIMULATION,
        fixtures={
            "storage": ANY,
            "ensemble": ANY,
            "reports_dir": ANY,
            "random_seed": ANY,
            "run_paths": ANY,
        },
    ),
    call(
        HookRuntime.POST_SIMULATION,
        fixtures={
            "storage": ANY,
            "ensemble": ANY,
            "reports_dir": ANY,
            "random_seed": ANY,
            "run_paths": ANY,
        },
    ),
    call(
        HookRuntime.POST_EXPERIMENT,
        fixtures={
            "random_seed": ANY,
            "storage": ANY,
            "ensemble": ANY,
        },
    ),
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
    monkeypatch.setattr(base_run_model, "_seed_sequence", MagicMock(return_value=0))
    monkeypatch.setattr(base_run_model.BaseRunModel, "run_workflows", run_wfs_mock)

    ens_mock = MagicMock()
    ens_mock.iteration = 0
    ens_mock.id = uuid.uuid1()
    storage_mock = MagicMock()
    storage_mock.create_ensemble.return_value = ens_mock

    test_class = EnsembleSmoother(
        *[MagicMock()] * 11,
    )
    test_class.run_ensemble_evaluator = MagicMock(return_value=[0])
    test_class._storage = storage_mock
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
    monkeypatch.setattr(base_run_model, "_seed_sequence", MagicMock(return_value=0))
    monkeypatch.setattr(base_run_model.BaseRunModel, "run_workflows", run_wfs_mock)

    ens_mock = MagicMock()
    ens_mock.iteration = 0
    ens_mock.id = uuid.uuid1()
    storage_mock = MagicMock()
    storage_mock.create_ensemble.return_value = ens_mock
    test_class = MultipleDataAssimilation(
        *[MagicMock()] * 14,
    )
    test_class._storage = storage_mock
    test_class.restart_run = False
    test_class.run_ensemble_evaluator = MagicMock(return_value=[0])
    test_class._design_matrix = None
    test_class.run_experiment(MagicMock())

    assert run_wfs_mock.mock_calls == EXPECTED_CALL_ORDER
