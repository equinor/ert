from unittest.mock import MagicMock, call

from ert._c_wrappers.enkf.enums import HookRuntime
from ert.shared.ensemble_evaluator.config import EvaluatorServerConfig
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
    ert_mock = MagicMock()
    minimum_args = MagicMock()
    test_class = EnsembleSmoother(minimum_args, ert_mock, MagicMock(), "experiment_id")
    test_class.create_context = MagicMock()
    test_class.run_ensemble_evaluator = MagicMock(return_value=1)
    ert_mock.runWorkflows = MagicMock()
    test_class.facade._es_update = MagicMock()
    evaluator_server_config_mock = MagicMock()
    test_class.runSimulations(evaluator_server_config_mock)

    expected_calls = [call(expected_call) for expected_call in EXPECTED_CALL_ORDER]
    assert ert_mock.runWorkflows.mock_calls == expected_calls


def test_hook_call_order_es_mda(monkeypatch):
    """
    The goal of this test is to assert that the hook call order is the same
    across different models.
    """
    minimum_args = {
        "start_iteration": 0,
        "weights": [1],
        "analysis_module": "some_module",
    }
    evaluator_server_config = EvaluatorServerConfig(
        custom_port_range=range(1024, 65535), use_token=False, generate_cert=False
    )
    ert_mock = MagicMock()
    test_class = MultipleDataAssimilation(
        minimum_args, ert_mock, MagicMock(), "experiment_id"
    )
    ert_mock.runWorkflows = MagicMock()

    test_class.create_context = MagicMock()
    test_class._checkMinimumActiveRealizations = MagicMock()
    test_class.parseWeights = MagicMock(return_value=[1])
    test_class.setAnalysisModule = MagicMock()
    test_class.facade.get_number_of_iterations = MagicMock(return_value=-1)
    test_class.facade._es_update = MagicMock()

    test_class.run_ensemble_evaluator = MagicMock(return_value=1)

    test_class.runSimulations(evaluator_server_config)

    expected_calls = [call(expected_call) for expected_call in EXPECTED_CALL_ORDER]
    assert ert_mock.runWorkflows.mock_calls == expected_calls


class MockWContainer:
    def __init__(self):
        self.iteration_nr = 1


class MockEsUpdate:
    def iterative_smoother_update(self, _, w_container):
        w_container.iteration_nr += 1


def test_hook_call_order_iterative_ensemble_smoother(monkeypatch):
    """
    The goal of this test is to assert that the hook call order is the same
    across different models.
    """
    ert_mock = MagicMock()
    ert_mock.analysisConfig.return_value.num_retries_per_iter = 1
    ert_mock.runWorkflows = MagicMock()

    test_class = IteratedEnsembleSmoother(
        MagicMock(), ert_mock, MagicMock(), "experiment_id"
    )
    test_class.create_context = MagicMock()
    test_class._checkMinimumActiveRealizations = MagicMock()
    test_class._ert = ert_mock
    test_class.parseWeights = MagicMock(return_value=[1])
    test_class.setAnalysisModule = MagicMock()
    test_class.run_ensemble_evaluator = MagicMock(return_value=1)

    test_class.setPhase = MagicMock()
    test_class.facade.get_number_of_iterations = MagicMock(return_value=1)
    test_class.facade._es_update = MockEsUpdate()
    test_class._w_container = MockWContainer()
    test_class.runSimulations(MagicMock())

    expected_calls = [call(expected_call) for expected_call in EXPECTED_CALL_ORDER]
    assert ert_mock.runWorkflows.mock_calls == expected_calls
