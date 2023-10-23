from unittest.mock import MagicMock
from uuid import UUID

import pytest

from ert.run_models import BaseRunModel
from ert.run_models.run_arguments import (
    EnsembleExperimentRunArguments,
    SimulationArguments,
)


@pytest.fixture
def base_arguments():
    return SimulationArguments(
        random_seed=1234, minimum_required_realizations=0, ensemble_size=1
    )


def test_base_run_model_supports_restart(minimum_case, base_arguments):
    ert = minimum_case
    BaseRunModel.validate = MagicMock()
    brm = BaseRunModel(
        base_arguments, ert, None, None, ert.get_queue_config(), UUID(int=0)
    )
    assert brm.support_restart


class MockJob:
    def __init__(self, status):
        self.status = status


@pytest.mark.parametrize(
    "initials, expected",
    [
        ([], []),
        ([True], [0]),
        ([False], []),
        ([False, True], [1]),
        ([True, True], [0, 1]),
        ([False, True], [1]),
    ],
)
def test_active_realizations(initials, expected, base_arguments):
    BaseRunModel.validate = MagicMock()
    brm = BaseRunModel(base_arguments, None, None, None, None, None)
    brm._initial_realizations_mask = initials
    assert brm._active_realizations == expected
    assert brm._ensemble_size == len(initials)


@pytest.mark.parametrize(
    "initials, completed, any_failed, failures",
    [
        ([True], [False], True, [True]),
        ([False], [False], False, [False]),
        ([False, True], [True, False], True, [False, True]),
        ([False, True], [False, True], False, [False, False]),
        ([False, False], [False, False], False, [False, False]),
        ([False, False], [True, True], False, [False, False]),
        ([True, True], [False, True], True, [True, False]),
        ([False, False], [], True, [True, True]),
    ],
)
def test_failed_realizations(initials, completed, any_failed, failures, base_arguments):
    BaseRunModel.validate = MagicMock()
    brm = BaseRunModel(base_arguments, None, None, None, None, None)
    brm._initial_realizations_mask = initials
    brm._completed_realizations_mask = completed

    assert brm._create_mask_from_failed_realizations() == failures
    assert brm._count_successful_realizations() == sum(completed)

    assert brm.has_failed_realizations() == any_failed


@pytest.mark.parametrize(
    "run_path, number_of_iterations, start_iteration, active_mask, expected",
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
    start_iteration: int,
    active_mask: list,
    expected: bool,
):
    simulation_arguments = EnsembleExperimentRunArguments(
        random_seed=None,
        active_realizations=active_mask,
        current_case=None,
        target_case=None,
        start_iteration=start_iteration,
        iter_num=0,
        minimum_required_realizations=0,
        ensemble_size=1,
    )

    def get_run_path_mock(realizations, iteration=None):
        if iteration is not None:
            return [f"out/realization-{r}/iter-{iteration}" for r in realizations]
        return [f"out/realization-{r}" for r in realizations]

    brm = BaseRunModel(simulation_arguments, None, None, None, None, None)
    brm.facade = MagicMock(
        run_path=run_path,
        number_of_iterations=number_of_iterations,
        get_run_paths=get_run_path_mock,
    )
    assert brm.check_if_runpath_exists() == expected
