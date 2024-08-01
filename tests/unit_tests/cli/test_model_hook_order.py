from unittest.mock import ANY, MagicMock, PropertyMock, call, patch

import pytest

from ert.config import HookRuntime
from ert.run_models import (
    BaseRunModel,
    EnsembleSmoother,
    IteratedEnsembleSmoother,
    MultipleDataAssimilation,
    base_run_model,
    ensemble_smoother,
    iterated_ensemble_smoother,
    multiple_data_assimilation,
)
from ert.run_models.run_arguments import (
    ESMDARunArguments,
    ESRunArguments,
    SIESRunArguments,
)

EXPECTED_CALL_ORDER = [
    HookRuntime.PRE_SIMULATION,
    HookRuntime.POST_SIMULATION,
    HookRuntime.PRE_FIRST_UPDATE,
    HookRuntime.PRE_UPDATE,
    HookRuntime.POST_UPDATE,
    HookRuntime.PRE_SIMULATION,
    HookRuntime.POST_SIMULATION,
]


@pytest.fixture
def patch_base_run_model(monkeypatch):
    monkeypatch.setattr(base_run_model, "create_run_path", MagicMock())
    monkeypatch.setattr(BaseRunModel, "validate", MagicMock())
    monkeypatch.setattr(BaseRunModel, "set_env_key", MagicMock())


@pytest.mark.usefixtures("patch_base_run_model")
def test_hook_call_order_ensemble_smoother(monkeypatch):
    """
    The goal of this test is to assert that the hook call order is the same
    across different models.
    """
    run_wfs_mock = MagicMock()
    monkeypatch.setattr(ensemble_smoother, "sample_prior", MagicMock())
    monkeypatch.setattr(ensemble_smoother, "smoother_update", MagicMock())
    monkeypatch.setattr(base_run_model, "LibresFacade", MagicMock())
    monkeypatch.setattr(base_run_model.BaseRunModel, "run_workflows", run_wfs_mock)
    monkeypatch.setattr(base_run_model.BaseRunModel, "setCurrentIteration", MagicMock())

    minimum_args = ESRunArguments(
        random_seed=None,
        active_realizations=[True],
        target_ensemble="smooth_%d",
        minimum_required_realizations=0,
        ensemble_size=1,
        experiment_name="no-name",
    )
    test_class = EnsembleSmoother(
        minimum_args,
        MagicMock(),
        MagicMock(),
        MagicMock(),
        MagicMock(),
        MagicMock(),
        MagicMock(),
    )
    test_class.run_ensemble_evaluator = MagicMock(return_value=[0])
    test_class.run_experiment(MagicMock())

    expected_calls = [
        call(expected_call, ANY, ANY) for expected_call in EXPECTED_CALL_ORDER
    ]
    assert run_wfs_mock.mock_calls == expected_calls


@pytest.mark.usefixtures("patch_base_run_model")
def test_hook_call_order_es_mda(monkeypatch):
    """
    The goal of this test is to assert that the hook call order is the same
    across different models.
    """

    minimum_args = ESMDARunArguments(
        random_seed=None,
        active_realizations=[True],
        target_ensemble="target_%d",
        weights="1",
        restart_run=False,
        prior_ensemble_id="",
        minimum_required_realizations=0,
        ensemble_size=1,
        experiment_name="no-name",
        starting_iteration=0,
    )
    run_wfs_mock = MagicMock()
    monkeypatch.setattr(multiple_data_assimilation, "sample_prior", MagicMock())
    monkeypatch.setattr(multiple_data_assimilation, "smoother_update", MagicMock())
    monkeypatch.setattr(base_run_model, "LibresFacade", MagicMock())
    monkeypatch.setattr(base_run_model.BaseRunModel, "run_workflows", run_wfs_mock)

    ens_mock = MagicMock()
    ens_mock.iteration = 0
    storage_mock = MagicMock()
    storage_mock.create_ensemble.return_value = ens_mock
    test_class = MultipleDataAssimilation(
        minimum_args,
        MagicMock(),
        storage_mock,
        MagicMock(),
        es_settings=MagicMock(),
        update_settings=MagicMock(),
        status_queue=MagicMock(),
    )
    test_class.run_ensemble_evaluator = MagicMock(return_value=[0])
    test_class.run_experiment(MagicMock())

    expected_calls = [
        call(expected_call, ANY, ANY) for expected_call in EXPECTED_CALL_ORDER
    ]
    assert run_wfs_mock.mock_calls == expected_calls


@pytest.mark.usefixtures("patch_base_run_model")
def test_hook_call_order_iterative_ensemble_smoother(monkeypatch):
    """
    The goal of this test is to assert that the hook call order is the same
    across different models.
    """
    run_wfs_mock = MagicMock()
    monkeypatch.setattr(iterated_ensemble_smoother, "sample_prior", MagicMock())
    monkeypatch.setattr(base_run_model, "LibresFacade", MagicMock())
    monkeypatch.setattr(base_run_model.BaseRunModel, "run_workflows", run_wfs_mock)
    monkeypatch.setattr(base_run_model.BaseRunModel, "setCurrentIteration", MagicMock())

    minimum_args = SIESRunArguments(
        random_seed=None,
        active_realizations=[True],
        target_ensemble="target_%d",
        number_of_iterations=1,
        num_retries_per_iter=1,
        minimum_required_realizations=0,
        ensemble_size=1,
        experiment_name="no-name",
    )
    test_class = IteratedEnsembleSmoother(
        minimum_args,
        MagicMock(),
        MagicMock(),
        MagicMock(),
        MagicMock(),
        MagicMock(),
        MagicMock(),
    )
    test_class.run_ensemble_evaluator = MagicMock(return_value=[0])

    # Mock the return values of iterative_smoother_update
    # Mock the iteration property of IteratedEnsembleSmoother
    with patch(
        "ert.run_models.iterated_ensemble_smoother.iterative_smoother_update",
        MagicMock(return_value=(MagicMock(), MagicMock())),
    ), patch(
        "ert.run_models.iterated_ensemble_smoother.IteratedEnsembleSmoother.sies_iteration",
        new_callable=PropertyMock,
    ) as mock_iteration:
        mock_iteration.return_value = 2
        test_class.run_experiment(MagicMock())

    expected_calls = [
        call(expected_call, ANY, ANY) for expected_call in EXPECTED_CALL_ORDER
    ]
    assert run_wfs_mock.mock_calls == expected_calls
