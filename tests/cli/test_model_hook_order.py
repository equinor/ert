from ert_shared.models import (
    EnsembleSmoother,
    MultipleDataAssimilation,
    IteratedEnsembleSmoother,
)
import inspect

from unittest.mock import MagicMock, call
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
    test_class = EnsembleSmoother
    mock_parent = MagicMock()
    mock_sim_runner = MagicMock()
    mock_parent.runWorkflows = mock_sim_runner
    ERT_mock = MagicMock()
    test_module = inspect.getmodule(test_class)
    monkeypatch.setattr(test_module, "EnkfSimulationRunner", mock_parent)
    monkeypatch.setattr(test_module, "ERT", ERT_mock)
    test_class.runSimulations(MagicMock(), ERT_mock)

    expected_calls = [
        call(expected_call, ert=ERT_mock.ert) for expected_call in EXPECTED_CALL_ORDER
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
    mock_sim_runner = MagicMock()
    mock_parent = MagicMock()
    mock_parent.runWorkflows = mock_sim_runner
    ERT_mock = MagicMock()

    test_module = inspect.getmodule(test_class)
    monkeypatch.setattr(test_module, "EnkfSimulationRunner", mock_parent)
    monkeypatch.setattr(test_module, "ERT", ERT_mock)

    test_class = test_class()
    test_class.create_context = MagicMock()
    test_class.checkMinimumActiveRealizations = MagicMock()
    test_class.parseWeights = MagicMock(return_value=[1])
    test_class.setAnalysisModule = MagicMock()
    test_class.ert = MagicMock()

    sim_runner_mock = MagicMock()
    sim_runner_mock.runSimpleStep.return_value = 1
    test_class.ert.return_value.getEnkfSimulationRunner = sim_runner_mock

    test_class.runSimulations(minimum_args)

    expected_calls = [
        call(expected_call, ert=ERT_mock.ert) for expected_call in EXPECTED_CALL_ORDER
    ]
    assert mock_sim_runner.mock_calls == expected_calls


def test_hook_call_order_iterative_ensemble_smoother(monkeypatch):
    """
    The goal of this test is to assert that the hook call order is the same
    across different models.
    """
    test_class = IteratedEnsembleSmoother
    minimum_args = MagicMock()
    mock_sim_runner = MagicMock()
    mock_parent = MagicMock()
    mock_parent.runWorkflows = mock_sim_runner

    ERT_mock = MagicMock()
    ERT_mock.enkf_facade.get_number_of_iterations.return_value = 1

    test_module = inspect.getmodule(test_class)
    monkeypatch.setattr(test_module, "EnkfSimulationRunner", mock_parent)
    monkeypatch.setattr(test_module, "ERT", ERT_mock)

    test_class = test_class()
    test_class.create_context = MagicMock()
    test_class.checkMinimumActiveRealizations = MagicMock()
    test_class.parseWeights = MagicMock(return_value=[1])
    test_class.setAnalysisModule = MagicMock()
    test_class.ert = MagicMock()

    analysis_config = MagicMock()
    analysis_config.getAnalysisIterConfig.return_value.getNumRetries.return_value = 1
    test_class.ert.return_value.analysisConfig.return_value = analysis_config

    simple_step_mock = MagicMock()
    simple_step_mock.runSimpleStep.return_value = 1
    test_class.ert.return_value.getEnkfSimulationRunner.return_value = simple_step_mock

    test_class.setAnalysisModule = MagicMock()
    test_class.setAnalysisModule.return_value.getInt.return_value = 1
    test_class.setPhase = MagicMock()

    test_class.runSimulations(minimum_args)

    expected_calls = [
        call(expected_call, ert=ERT_mock.ert) for expected_call in EXPECTED_CALL_ORDER
    ]
    assert mock_sim_runner.mock_calls == expected_calls
