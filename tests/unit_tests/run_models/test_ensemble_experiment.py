from unittest.mock import MagicMock

import pytest

from ert.run_models import EnsembleExperiment
from ert.run_models.run_arguments import (
    EnsembleExperimentRunArguments,
)


@pytest.mark.parametrize(
    "iter_num, active_mask, expected",
    [
        (2, [True, True, True, True], False),
        (1, [True, True, True, True], True),
        (1, [False, False, True, False], False),
        (0, [False, False, False, False], False),
        (0, [], False),
        (1, [False, True, True], True),
        (0, [False, False, True], False),
    ],
)
def test_check_if_runpath_exists(
    create_dummy_run_path,
    iter_num: int,
    active_mask: list,
    expected: bool,
):
    simulation_arguments = EnsembleExperimentRunArguments(
        random_seed=None,
        active_realizations=active_mask,
        current_ensemble=None,
        target_ensemble=None,
        iter_num=iter_num,
        minimum_required_realizations=0,
        ensemble_size=1,
        stop_long_running=False,
        experiment_name="no-name",
    )

    def get_run_path_mock(realizations, iteration=None):
        if iteration is not None:
            return [f"out/realization-{r}/iter-{iteration}" for r in realizations]
        return [f"out/realization-{r}" for r in realizations]

    EnsembleExperiment.validate = MagicMock()
    ensemble_experiment = EnsembleExperiment(
        simulation_arguments, MagicMock(), None, None, MagicMock()
    )
    ensemble_experiment.run_paths.get_paths = get_run_path_mock

    assert ensemble_experiment.check_if_runpath_exists() == expected
