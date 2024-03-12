from unittest.mock import MagicMock

import pytest

from ert.run_models import (
    MultipleDataAssimilation,
    base_run_model,
    multiple_data_assimilation,
)
from ert.run_models.run_arguments import ESMDARunArguments
from ert.storage import Ensemble


@pytest.mark.usefixtures("use_tmpdir")
def test_that_all_iterations_gets_correct_name_and_iteration_number(
    storage, monkeypatch
):
    minimum_args = ESMDARunArguments(
        random_seed=None,
        active_realizations=[True],
        target_ensemble="target_%d",
        weights="1, 2, 3",
        restart_run=False,
        prior_ensemble="",
        minimum_required_realizations=1,
        ensemble_size=1,
        stop_long_running=True,
        experiment_name="no-name",
    )
    ens_mock = MagicMock()
    ens_mock.iteration = 0
    context_mock = MagicMock()
    monkeypatch.setattr(multiple_data_assimilation, "RunContext", context_mock)
    monkeypatch.setattr(base_run_model, "LibresFacade", MagicMock())
    monkeypatch.setattr(MultipleDataAssimilation, "validate", MagicMock())
    monkeypatch.setattr(MultipleDataAssimilation, "setPhase", MagicMock())
    monkeypatch.setattr(MultipleDataAssimilation, "set_env_key", MagicMock())
    monkeypatch.setattr(multiple_data_assimilation, "smoother_update", MagicMock())
    monkeypatch.setattr(base_run_model, "EnKFMain", MagicMock())

    test_class = MultipleDataAssimilation(
        minimum_args,
        MagicMock(),
        storage,
        MagicMock(),
        es_settings=MagicMock(),
        update_settings=MagicMock(),
        status_queue=MagicMock(),
    )
    test_class.run_ensemble_evaluator = MagicMock(return_value=[0])
    test_class.run_experiment(MagicMock())

    # Find all the ensemble_context calls and fetch the ensemble name and
    # iteration number
    calls = set(
        (x.kwargs["ensemble"].name, x.kwargs["iteration"])
        for x in context_mock.mock_calls
        if "ensemble" in x.kwargs and isinstance(x.kwargs["ensemble"], Ensemble)
    )
    assert ("target_0", 0) in calls
    assert ("target_1", 1) in calls
    assert ("target_2", 2) in calls
