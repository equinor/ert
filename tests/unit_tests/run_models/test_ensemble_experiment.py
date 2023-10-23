from unittest.mock import MagicMock

import pytest

from ert.run_models import EnsembleExperiment
from ert.run_models.run_arguments import (
    EnsembleExperimentRunArguments,
)


@pytest.mark.parametrize(
    "run_path, number_of_iterations, iter_num, active_mask, expected",
    [
        ("out/realization-%d/iter-%d", 4, 2, [True, True, True, True], False),
        ("out/realization-%d/iter-%d", 4, 1, [True, True, True, True], True),
        ("out/realization-%d/iter-%d", 4, 1, [False, False, True, False], False),
        ("out/realization-%d/iter-%d", 4, 0, [False, False, False, False], False),
        ("out/realization-%d/iter-%d", 4, 0, [], False),
        ("out/realization-%d", 2, 1, [False, True, True], True),
        ("out/realization-%d", 2, 0, [False, False, True], False),
    ],
)
def test_check_if_runpath_exists(
    create_dummy_run_path,
    run_path: str,
    number_of_iterations: int,
    iter_num: int,
    active_mask: list,
    expected: bool,
):
    simulation_arguments = EnsembleExperimentRunArguments(
        random_seed=None,
        active_realizations=active_mask,
        current_case=None,
        target_case=None,
        iter_num=iter_num,
        minimum_required_realizations=0,
        ensemble_size=1,
    )

    def get_run_path_mock(realizations, iteration=None):
        if iteration is not None:
            return [f"out/realization-{r}/iter-{iteration}" for r in realizations]
        return [f"out/realization-{r}" for r in realizations]

    ensemble_experiment = EnsembleExperiment(
        simulation_arguments, None, None, None, None
    )

    ensemble_experiment.facade = MagicMock(
        run_path=run_path,
        number_of_iterations=number_of_iterations,
        get_run_paths=get_run_path_mock,
    )

    assert ensemble_experiment.check_if_runpath_exists() == expected
