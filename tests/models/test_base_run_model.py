import sys
import unittest

from ert_gui.ertnotifier import configureErtNotifier
from ert_shared.models import BaseRunModel
from res.job_queue import JobStatusType
from res.test import ErtTestContext
from utils import ErtTest

from unittest.mock import Mock, patch


class BaseRunModelTest(ErtTest):
    def test_instantiation(self):
        config_file = self.createTestPath("local/simple_config/minimum_config")
        with ErtTestContext("kjell", config_file) as work_area:
            ert = work_area.getErt()
            configureErtNotifier(ert, config_file)
            brm = BaseRunModel(ert.get_queue_config())
            self.assertFalse(brm.isQueueRunning())


class InMemoryBaseRunModelTest(unittest.TestCase):
    def test_detailed_progress(self):
        # TODO: rewrite to make use of fixtures
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

        with patch("ert_shared.models.base_run_model.ForwardModelStatus") as f:
            f.load.return_value = Mock()
            f.load.return_value.jobs = [{"name": "job1"}]
            brm.updateDetailedProgress()

        jobs, status = brm.realization_progress[0][0]
        self.assertEqual(len(jobs), 1)
        self.assertIn("name", jobs[0])
