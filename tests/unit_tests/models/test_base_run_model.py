import os
from unittest.mock import MagicMock

import pytest

from ert._c_wrappers.enkf import EnKFMain
from ert._c_wrappers.job_queue import RunStatusType
from ert.shared.models import BaseRunModel


def test_base_run_model_supports_restart(setup_case):
    ert = EnKFMain(setup_case("simple_config/", "minimum_config"))
    brm = BaseRunModel(None, ert, ert.get_queue_config(), "experiment_id")
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
def test_active_realizations(initials, expected):
    brm = BaseRunModel(None, None, None, None)
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
def test_failed_realizations(initials, completed, any_failed, failures):
    brm = BaseRunModel(None, None, None, None)
    brm._initial_realizations_mask = initials
    brm._completed_realizations_mask = completed

    assert brm._create_mask_from_failed_realizations() == failures
    assert brm._count_successful_realizations() == sum(completed)

    assert brm.has_failed_realizations() == any_failed


@pytest.mark.usefixtures("use_tmpdir")
def test_run_ensemble_evaluator():
    run_arg = MagicMock()
    run_arg.run_status = RunStatusType.JOB_LOAD_FAILURE
    run_context = MagicMock()
    run_context.__iter__.return_value = [run_arg]
    run_context.is_active.return_value = True
    BaseRunModel.deactivate_failed_jobs(run_context)

    run_context.deactivate_realization.assert_called_with(0)


@pytest.fixture
def create_dummy_run_path(tmpdir):
    run_path = os.path.join(tmpdir, "out")
    os.mkdir(run_path)
    os.mkdir(os.path.join(run_path, "realization-0"))
    os.mkdir(os.path.join(run_path, "realization-0/iter-0"))
    os.mkdir(os.path.join(run_path, "realization-1"))
    os.mkdir(os.path.join(run_path, "realization-1/iter-0"))
    os.mkdir(os.path.join(run_path, "realization-1/iter-1"))
    yield os.chdir(tmpdir)


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
    simulation_arguments = {
        "start_iteration": start_iteration,
        "active_realizations": active_mask,
    }

    def get_run_path_mock(realizations, iteration=None):
        if iteration is not None:
            return [f"out/realization-{r}/iter-{iteration}" for r in realizations]
        return [f"out/realization-{r}" for r in realizations]

    brm = BaseRunModel(simulation_arguments, None, None, None)
    brm.facade = MagicMock(
        run_path=run_path,
        number_of_iterations=number_of_iterations,
        get_run_paths=get_run_path_mock,
    )

    assert brm.check_if_runpath_exists() == expected


def test_validation():
    simulation_arguments = {
        "start_iteration": 0,
        "active_realizations": list(range(50)),
        "current_case": "something",
    }
    ert = MagicMock(
        _ensemble_size=100,
        storage_manager=MagicMock(
            __get_item__=lambda _: MagicMock(state_map=list(range(10))),
            __contains__=lambda _, __: True,
        ),
    )
    with pytest.raises(ValueError):
        BaseRunModel(simulation_arguments, ert, None, None, None)
