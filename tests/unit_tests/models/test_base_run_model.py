from unittest.mock import MagicMock

import pytest

from ert._c_wrappers.enkf import EnkfFs, EnKFMain, RunContext
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

    run_context = RunContext(
        EnkfFs.createFileSystem("test", True, 10),
        mask=[True],
        paths=["some%d/path%d"],
        jobnames=["some_job%d"],
        iteration=0,
    )
    run_context.run_args = [run_arg]
    run_context.deactivate_realization = MagicMock()

    BaseRunModel.deactivate_failed_jobs(run_context)

    run_context.deactivate_realization.assert_called_with(0)
