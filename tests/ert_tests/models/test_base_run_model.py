from unittest.mock import MagicMock

import pytest
from ert_utils import ErtTest

from ert_shared.models import BaseRunModel
from res.enkf import ErtRunContext
from res.job_queue import RunStatusType
from res.test import ErtTestContext


class BaseRunModelTest(ErtTest):
    def test_instantiation(self):
        config_file = self.createTestPath("local/simple_config/minimum_config")
        with ErtTestContext(config_file) as work_area:
            ert = work_area.getErt()
            brm = BaseRunModel(None, ert, ert.get_queue_config())
            assert brm.support_restart


class MockJob:
    def __init__(self, status):
        self.status = status


"""
NOTE: The test below is for the old job-status mechanism no
longer used in ERT. The mechanism is, however, used in Everest
so we leave this test in place.

If Everest at some point updates how it tracks job-status, this
test as well as BaseRunModel.is_forward_model_finished() can be
removed.
"""


@pytest.mark.parametrize(
    "test_input, expected",
    [
        ([MockJob("Success")], True),
        ([MockJob("Failure")], False),
        ([MockJob("Success"), MockJob("Success")], True),
        ([MockJob("Failure"), MockJob("Success")], False),
    ],
)
def test_is_forward_model_finished(test_input, expected):
    assert BaseRunModel.is_forward_model_finished(test_input) is expected


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
    brm = BaseRunModel(None, None, None)
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
    ],
)
def test_failed_realizations(initials, completed, any_failed, failures):
    brm = BaseRunModel(None, None, None)
    brm._initial_realizations_mask = initials
    brm._completed_realizations_mask = completed

    assert brm._create_mask_from_failed_realizations() == failures
    assert brm._count_successful_realizations() == sum(completed)

    assert brm.has_failed_realizations() == any_failed


def test_run_ensemble_evaluator():
    run_arg = MagicMock()
    run_arg.run_status = RunStatusType.JOB_LOAD_FAILURE

    run_context = ErtRunContext.ensemble_experiment(
        None, [True], ["some%d/path%d"], ["some_job%d"], 0
    )
    run_context._iget = MagicMock()
    run_context._iget.return_value = run_arg
    run_context._deactivate_realization = MagicMock()

    BaseRunModel.deactivate_failed_jobs(run_context)

    run_context._deactivate_realization.assert_called_with(0)
