import os
from pathlib import Path
from unittest.mock import MagicMock
from uuid import UUID

import pytest

from ert.config import ModelConfig
from ert.run_models import BaseRunModel
from ert.run_models.run_arguments import (
    EnsembleExperimentRunArguments,
    SimulationArguments,
)
from ert.substitution_list import SubstitutionList


@pytest.fixture
def base_arguments():
    return SimulationArguments(
        random_seed=1234,
        minimum_required_realizations=0,
        ensemble_size=1,
        stop_long_running=False,
    )


def test_base_run_model_supports_restart(minimum_case, base_arguments):
    BaseRunModel.validate = MagicMock()
    brm = BaseRunModel(
        base_arguments, minimum_case, None, None, minimum_case.queue_config, UUID(int=0)
    )
    assert brm.support_restart


class MockJob:
    def __init__(self, status):
        self.status = status


@pytest.mark.parametrize(
    "initials",
    [
        ([]),
        ([True]),
        ([False]),
        ([False, True]),
        ([True, True]),
        ([False, True]),
    ],
)
def test_active_realizations(initials, base_arguments):
    BaseRunModel.validate = MagicMock()
    brm = BaseRunModel(base_arguments, MagicMock(), None, None, None, None)
    brm._initial_realizations_mask = initials
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
    brm = BaseRunModel(base_arguments, MagicMock(), None, None, None, None)
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
        stop_long_running=False,
    )
    simulation_arguments.num_iterations = number_of_iterations

    def get_run_path_mock(realizations, iteration=None):
        if iteration is not None:
            return [f"out/realization-{r}/iter-{iteration}" for r in realizations]
        return [f"out/realization-{r}" for r in realizations]

    brm = BaseRunModel(simulation_arguments, MagicMock(), None, None, None, None)
    brm.run_paths.get_paths = get_run_path_mock
    brm.facade = MagicMock(
        run_path=run_path,
    )
    assert brm.check_if_runpath_exists() == expected


@pytest.mark.usefixtures("use_tmpdir")
@pytest.mark.parametrize(
    "run_path_format", ["realization-<IENS>/iter-<ITER>", "realization-<IENS>"]
)
@pytest.mark.parametrize(
    "active_realizations", [[True], [True, True], [True, False], [False], [False, True]]
)
def test_delete_run_path(run_path_format, active_realizations):
    simulation_arguments = EnsembleExperimentRunArguments(
        random_seed=None,
        active_realizations=active_realizations,
        current_case=None,
        target_case=None,
        start_iteration=0,
        iter_num=0,
        minimum_required_realizations=0,
        ensemble_size=1,
        stop_long_running=False,
    )
    expected_remaining = []
    expected_removed = []
    for iens, mask in enumerate(active_realizations):
        run_path = Path(
            run_path_format.replace("<IENS>", str(iens)).replace("<ITER>", "0")
        )
        os.makedirs(run_path)
        assert run_path.exists()
        if not mask:
            expected_remaining.append(run_path)
        else:
            expected_removed.append(run_path)
    share_path = Path("share")
    os.makedirs(share_path)
    model_config = ModelConfig(runpath_format_string=run_path_format)
    subs_list = SubstitutionList()
    config = MagicMock()
    config.model_config = model_config
    config.substitution_list = subs_list

    brm = BaseRunModel(simulation_arguments, config, None, None, None, None)
    brm.rm_run_path()
    assert not any(path.exists() for path in expected_removed)
    assert all(path.parent.exists() for path in expected_removed)
    assert all(path.exists() for path in expected_remaining)
    assert share_path.exists()
