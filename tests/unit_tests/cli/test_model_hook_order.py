from unittest.mock import ANY, MagicMock, call, patch
from uuid import UUID

import numpy as np
import pytest

from ert.config import HookRuntime
from ert.run_models import (
    EnsembleSmoother,
    IteratedEnsembleSmoother,
    MultipleDataAssimilation,
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


@pytest.mark.usefixtures("use_tmpdir")
def test_hook_call_order_ensemble_smoother(storage):
    """
    The goal of this test is to assert that the hook call order is the same
    across different models.
    """
    ert_mock = MagicMock(
        _ensemble_size=0,
        analysisConfig=lambda: MagicMock(minimum_required_realizations=0),
    )
    ert_mock.ensemble_context.return_value = MagicMock(iteration=0)
    ert_mock.ensemble_context.return_value.sim_fs.get_realization_mask_from_state = (
        MagicMock(return_value=np.array([True]))
    )

    minimum_args = ESRunArguments(
        active_realizations=[True], current_case="default", target_case="smooth"
    )
    test_class = EnsembleSmoother(
        minimum_args, ert_mock, storage, MagicMock(), UUID(int=0)
    )
    test_class.run_ensemble_evaluator = MagicMock(return_value=1)
    test_class.run_experiment(MagicMock())

    expected_calls = [
        call(expected_call, ANY, ANY) for expected_call in EXPECTED_CALL_ORDER
    ]
    assert ert_mock.runWorkflows.mock_calls == expected_calls


@pytest.mark.usefixtures("use_tmpdir")
def test_hook_call_order_es_mda(monkeypatch):
    """
    The goal of this test is to assert that the hook call order is the same
    across different models.
    """

    minimum_args = ESMDARunArguments(
        active_realizations=[True],
        target_case="target_%d",
        weights="1",
        restart_run=False,
        prior_ensemble="",
    )

    ert_mock = MagicMock(
        analysisConfig=lambda: MagicMock(minimum_required_realizations=0),
    )
    ert_mock.ensemble_context.return_value = MagicMock(iteration=1)
    ert_mock.ensemble_context.return_value.sim_fs.get_realization_mask_from_state = (
        MagicMock(return_value=np.array([True]))
    )
    monkeypatch.setattr(MultipleDataAssimilation, "validate", MagicMock())
    ens_mock = MagicMock()
    ens_mock.iteration = 0
    storage_mock = MagicMock()
    storage_mock.create_ensemble.return_value = ens_mock
    test_class = MultipleDataAssimilation(
        minimum_args,
        ert_mock,
        storage_mock,
        MagicMock(),
        UUID(int=0),
        prior_ensemble=None,
    )
    ert_mock.runWorkflows = MagicMock()
    test_class.run_ensemble_evaluator = MagicMock(return_value=1)
    test_class.run_experiment(MagicMock())

    expected_calls = [
        call(expected_call, ANY, ANY) for expected_call in EXPECTED_CALL_ORDER
    ]
    assert ert_mock.runWorkflows.mock_calls == expected_calls


class MockWContainer:
    def __init__(self):
        self.iteration_nr = 1


def mock_iterative_smoother_update(_, posterior_storage, w_container, *args, **kwargs):
    w_container.iteration_nr += 1


@pytest.mark.usefixtures("use_tmpdir")
def test_hook_call_order_iterative_ensemble_smoother(monkeypatch):
    """
    The goal of this test is to assert that the hook call order is the same
    across different models.
    """
    ert_mock = MagicMock(
        _ensemble_size=10,
        analysisConfig=lambda: MagicMock(minimum_required_realizations=0),
    )
    ert_mock.ensemble_context.return_value.iteration = 1
    ert_mock.ensemble_context.return_value.sim_fs.get_realization_mask_from_state = (
        MagicMock(return_value=np.array([True]))
    )
    minimum_args = SIESRunArguments(
        active_realizations=[True],
        current_case="default",
        target_case="target_%d",
        num_iterations=1,
    )
    monkeypatch.setattr(IteratedEnsembleSmoother, "validate", MagicMock())
    test_class = IteratedEnsembleSmoother(
        minimum_args, ert_mock, MagicMock(), MagicMock(), UUID(int=0)
    )
    test_class.run_ensemble_evaluator = MagicMock(return_value=1)

    test_class.facade.get_number_of_iterations = MagicMock(return_value=1)
    test_class._w_container = MockWContainer()

    with patch(
        "ert.run_models.iterated_ensemble_smoother.iterative_smoother_update",
        mock_iterative_smoother_update,
    ):
        test_class.run_experiment(MagicMock())

    expected_calls = [
        call(expected_call, ANY, ANY) for expected_call in EXPECTED_CALL_ORDER
    ]
    assert ert_mock.runWorkflows.mock_calls == expected_calls
