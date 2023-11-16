from unittest.mock import MagicMock
from uuid import UUID

import pytest

from ert.run_models import (
    MultipleDataAssimilation,
    base_run_model,
    multiple_data_assimilation,
)
from ert.run_models.run_arguments import ESMDARunArguments
from ert.storage import EnsembleAccessor


@pytest.mark.usefixtures("use_tmpdir")
def test_that_all_iterations_gets_correct_name_and_iteration_number(
    storage, monkeypatch
):
    minimum_args = ESMDARunArguments(
        random_seed=None,
        active_realizations=[True],
        target_case="target_%d",
        weights="1, 2, 3",
        restart_run=False,
        prior_ensemble="",
        minimum_required_realizations=1,
        ensemble_size=1,
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
        UUID(int=0),
        prior_ensemble=None,
        es_settings=MagicMock(),
        update_settings=MagicMock(),
    )
    test_class.run_ensemble_evaluator = MagicMock(return_value=1)
    test_class.run_experiment(MagicMock())

    # Find all the ensemble_context calls and fetch the ensemble name and
    # iteration number
    calls = set(
        (x.kwargs["sim_fs"].name, x.kwargs["iteration"])
        for x in context_mock.mock_calls
        if "sim_fs" in x.kwargs and isinstance(x.kwargs["sim_fs"], EnsembleAccessor)
    )
    assert ("target_0", 0) in calls
    assert ("target_1", 1) in calls
    assert ("target_2", 2) in calls
