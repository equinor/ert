from unittest.mock import Mock, patch
import pytest

from ert_utils import ErtTest

from ert_gui.ertnotifier import ErtNotifier
from ert_shared import ERT
from ert_shared.models import BaseRunModel
from res.job_queue import JobStatusType
from res.test import ErtTestContext


class BaseRunModelTest(ErtTest):
    def test_instantiation(self):
        config_file = self.createTestPath("local/simple_config/minimum_config")
        with ErtTestContext("kjell", config_file) as work_area:
            ert = work_area.getErt()
            notifier = ErtNotifier(ert, config_file)
            with ERT.adapt(notifier):
                brm = BaseRunModel(ert.get_queue_config())
                assert not brm.isQueueRunning()


def test_detailed_progress():
    brm = BaseRunModel(None)
    brm._run_context = Mock()
    brm._run_context.get_iter.return_value = 0

    run_arg1 = Mock()
    run_arg1.getQueueIndex.return_value = 0
    run_arg2 = Mock()
    run_arg2.getQueueIndex.return_value = 1
    run_arg2.iens = 0
    brm._run_context.__iter__ = Mock()
    brm._run_context.__iter__.return_value = iter([run_arg1, run_arg2])

    def job_status(queue_index):
        if queue_index == 0:
            return JobStatusType.JOB_QUEUE_PENDING
        if queue_index == 1:
            return JobStatusType.JOB_QUEUE_RUNNING

    brm._job_queue = Mock()
    brm._job_queue.getJobStatus.side_effect = job_status
    brm._job_queue.did_job_time_out.return_value = False

    with patch("ert_shared.models.base_run_model.ForwardModelStatus") as f:
        f.load.return_value = Mock()
        f.load.return_value.jobs = [{"name": "job1"}]
        brm.updateDetailedProgress()

    jobs, _ = brm.realization_progress[0][0]
    assert len(jobs) == 1
    assert "name" in jobs[0]


class MockJob:
    def __init__(self, status):
        self.status = status


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
