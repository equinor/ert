import inspect
from unittest.mock import MagicMock, call

from ert_shared.ensemble_evaluator.config import EvaluatorServerConfig
from ert_shared.models import (
    EnsembleSmoother,
    IteratedEnsembleSmoother,
    MultipleDataAssimilation,
)
from res.enkf.enums import HookRuntime

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
    test_class = EnsembleSmoother(minimum_args, ert_mock, MagicMock())
    test_class.create_context = MagicMock()
    test_class.run_ensemble_evaluator = MagicMock(return_value=1)
    mock_parent = MagicMock()
    mock_sim_runner = MagicMock()
    mock_parent.runWorkflows = mock_sim_runner
    evaluator_server_config_mock = MagicMock()
    test_module = inspect.getmodule(test_class)
    monkeypatch.setattr(test_module, "EnkfSimulationRunner", mock_parent)
    test_class.runSimulations(evaluator_server_config_mock)

    expected_calls = [
        call(expected_call, ert=ert_mock) for expected_call in EXPECTED_CALL_ORDER
    ]
    assert mock_sim_runner.mock_calls == expected_calls


def test_hook_call_order_es_mda(monkeypatch):
    """
    The goal of this test is to assert that the hook call order is the same
    across different models.
    """
    test_class = MultipleDataAssimilation
    minimum_args = {
        "start_iteration": 0,
        "weights": [1],
        "analysis_module": "some_module",
    }
    evaluator_server_config = EvaluatorServerConfig(
        custom_port_range=range(1024, 65535)
    )
    ert_mock = MagicMock()
    test_class = test_class(minimum_args, ert_mock, MagicMock())
    mock_sim_runner = MagicMock()
    mock_parent = MagicMock()
    mock_parent.runWorkflows = mock_sim_runner
    test_module = inspect.getmodule(test_class)
    monkeypatch.setattr(test_module, "EnkfSimulationRunner", mock_parent)

    test_class.create_context = MagicMock()
    test_class._checkMinimumActiveRealizations = MagicMock()
    test_class.parseWeights = MagicMock(return_value=[1])
    test_class.setAnalysisModule = MagicMock()
    test_class.facade.get_number_of_iterations = MagicMock(return_value=-1)

    test_class.run_ensemble_evaluator = MagicMock(return_value=1)

    test_class.runSimulations(evaluator_server_config)

    expected_calls = [
        call(expected_call, ert=ert_mock) for expected_call in EXPECTED_CALL_ORDER
    ]
    assert mock_sim_runner.mock_calls == expected_calls


def test_hook_call_order_iterative_ensemble_smoother(monkeypatch):
    """
    The goal of this test is to assert that the hook call order is the same
    across different models.
    """
    test_class = IteratedEnsembleSmoother

    mock_sim_runner = MagicMock()
    mock_parent = MagicMock()
    mock_parent.runWorkflows = mock_sim_runner

    test_module = inspect.getmodule(test_class)
    monkeypatch.setattr(test_module, "EnkfSimulationRunner", mock_parent)

    test_class.create_context = MagicMock()
    test_class._checkMinimumActiveRealizations = MagicMock()
    test_class.parseWeights = MagicMock(return_value=[1])
    test_class.setAnalysisModule = MagicMock()
    ert_mock = MagicMock()
    ert_mock.return_value.analysisConfig.return_value.getAnalysisIterConfig.return_value.getNumRetries.return_value = (  # noqa
        1
    )
    test_class = test_class(MagicMock(), MagicMock(), MagicMock())
    monkeypatch.setattr(test_class, "ert", ert_mock)
    test_class.setAnalysisModule = MagicMock()
    test_class.setAnalysisModule.return_value.getInt.return_value = 1
    test_class.run_ensemble_evaluator = MagicMock(return_value=1)

    test_class.setPhase = MagicMock()
    test_class.facade.get_number_of_iterations = MagicMock(return_value=1)
    test_class.runSimulations(MagicMock())

    expected_calls = [
        call(expected_call, ert=ert_mock()) for expected_call in EXPECTED_CALL_ORDER
    ]
    assert mock_sim_runner.mock_calls == expected_calls
