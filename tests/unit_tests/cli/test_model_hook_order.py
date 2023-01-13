from unittest.mock import MagicMock, call

from ert._c_wrappers.enkf.enums import HookRuntime
from ert.shared.models import (
    EnsembleSmoother,
    IteratedEnsembleSmoother,
    MultipleDataAssimilation,
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


def test_hook_call_order_ensemble_smoother(monkeypatch):
    """
    The goal of this test is to assert that the hook call order is the same
    across different models.
    """
    ert_mock = MagicMock(
        _ensemble_size=0,
        storage_manager=MagicMock(__get_item__=lambda x: MagicMock(state_map=[])),
    )
    minimum_args = {
        "current_case": "default",
        "active_realizations": [True],
        "target_case": "smooth",
    }
    test_class = EnsembleSmoother(minimum_args, ert_mock, MagicMock(), "experiment_id")
    test_class.run_ensemble_evaluator = MagicMock(return_value=1)
    test_class.runSimulations(MagicMock())

    expected_calls = [call(expected_call) for expected_call in EXPECTED_CALL_ORDER]
    assert ert_mock.runWorkflows.mock_calls == expected_calls


def test_hook_call_order_es_mda(monkeypatch):
    """
    The goal of this test is to assert that the hook call order is the same
    across different models.
    """
    minimum_args = {
        "start_iteration": 0,
        "weights": "1",
        "num_iterations": 1,
        "analysis_module": "some_module",
        "active_realizations": [True],
        "target_case": "target_%d",
    }
    ert_mock = MagicMock()
    test_class = MultipleDataAssimilation(
        minimum_args, ert_mock, MagicMock(), "experiment_id"
    )
    ert_mock.runWorkflows = MagicMock()
    test_class.run_ensemble_evaluator = MagicMock(return_value=1)
    test_class.runSimulations(MagicMock())

    expected_calls = [call(expected_call) for expected_call in EXPECTED_CALL_ORDER]
    assert ert_mock.runWorkflows.mock_calls == expected_calls


class MockWContainer:
    def __init__(self):
        self.iteration_nr = 1


class MockEsUpdate:
    def iterative_smoother_update(self, _, posterior_storage, w_container, run_id):
        w_container.iteration_nr += 1


def test_hook_call_order_iterative_ensemble_smoother(monkeypatch):
    """
    The goal of this test is to assert that the hook call order is the same
    across different models.
    """
    ert_mock = MagicMock(
        _ensemble_size=10,
        storage_manager=MagicMock(
            __get_item__=lambda x: MagicMock(state_map=list(range(10)))
        ),
    )
    minimum_args = {
        "num_iterations": 1,
        "active_realizations": [True],
        "target_case": "target_%d",
    }

    test_class = IteratedEnsembleSmoother(
        minimum_args, ert_mock, MagicMock(), "experiment_id"
    )
    test_class.run_ensemble_evaluator = MagicMock(return_value=1)

    test_class.setPhase = MagicMock()
    test_class.facade.get_number_of_iterations = MagicMock(return_value=1)
    test_class.facade._es_update = MockEsUpdate()
    test_class._w_container = MockWContainer()
    test_class.runSimulations(MagicMock())

    expected_calls = [call(expected_call) for expected_call in EXPECTED_CALL_ORDER]
    assert ert_mock.runWorkflows.mock_calls == expected_calls
