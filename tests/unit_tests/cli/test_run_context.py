from unittest.mock import MagicMock
from uuid import UUID

from ert.run_models import MultipleDataAssimilation
from ert.storage import EnsembleAccessor


def test_that_all_iterations_gets_correct_name_and_iteration_number(storage):
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
    ert_mock.ensemble_context.return_value.sim_fs.id = UUID(int=0)

    test_class = MultipleDataAssimilation(
        minimum_args, ert_mock, storage, MagicMock(), UUID(int=0), prior_ensemble=None
    )
    test_class.run_ensemble_evaluator = MagicMock(return_value=1)
    test_class.runSimulations(MagicMock())

    # Find all the ensemble_context calls and fetch the ensemble name and
    # iteration number
    calls = set(
        (x.args[0].name, x.kwargs["iteration"])
        for x in ert_mock.ensemble_context.mock_calls
        if len(x.args) > 0 and isinstance(x.args[0], EnsembleAccessor)
    )
    assert ("target_0", 0) in calls
    assert ("target_1", 1) in calls
    assert ("target_2", 2) in calls
