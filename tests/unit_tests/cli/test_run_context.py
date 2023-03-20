from unittest.mock import ANY, MagicMock, call

from ert.shared.models import MultipleDataAssimilation


def test_that_all_iterations_gets_correct_name_and_iteration_number():
    minimum_args = {
        "start_iteration": 0,
        "weights": "1, 2, 3",
        "num_iterations": 3,
        "analysis_module": "some_module",
        "active_realizations": [True],
        "target_case": "target_%d",
        "restart_run": False,
        "prior_ensemble": "",
    }
    ert_mock = MagicMock(
        analysisConfig=lambda: MagicMock(minimum_required_realizations=0),
    )
    test_class = MultipleDataAssimilation(
        minimum_args, ert_mock, MagicMock(), "experiment_id"
    )
    ert_mock.create_ensemble_context.return_value = MagicMock()
    test_class.run_ensemble_evaluator = MagicMock(return_value=1)
    test_class.runSimulations(MagicMock())

    calls = ert_mock.create_ensemble_context.mock_calls
    assert call("target_0", ANY, iteration=0) in calls
    assert call("target_1", ANY, iteration=1) in calls
    assert call("target_2", ANY, iteration=2) in calls
