from ert.job_queue import WorkflowJobMonitor
from ert.test import ExtendedTestCase


class WorkflowJobMonitorTest(ExtendedTestCase):

    def test_workflow_job_monitor_creation(self):
        monitor = WorkflowJobMonitor()

        monitor.setPID(55)
        self.assertEqual(monitor.getPID(), 55)

        self.assertTrue(monitor.isBlocking())

        monitor.setBlocking(False)
        self.assertFalse(monitor.isBlocking())


