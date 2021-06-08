import os
import time
from res.job_queue import WorkflowJoblist, Workflow, WorkflowRunner
from tests import ResTest
from tests.utils import wait_until
from ecl.util.test import TestAreaContext
from res.util.substitution_list import SubstitutionList
from .workflow_common import WorkflowCommon
import sys
from unittest.mock import patch


class WorkflowRunnerTest(ResTest):
    def test_workflow_thread_cancel_ert_script(self):
        with TestAreaContext(
            "python/job_queue/workflow_runner_ert_script"
        ) as work_area:
            WorkflowCommon.createWaitJob()

            joblist = WorkflowJoblist()
            self.assertTrue(joblist.addJobFromFile("WAIT", "wait_job"))
            self.assertTrue("WAIT" in joblist)

            workflow = Workflow("wait_workflow", joblist)

            self.assertEqual(len(workflow), 3)

            workflow_runner = WorkflowRunner(workflow)

            self.assertFalse(workflow_runner.isRunning())

            with workflow_runner:
                self.assertIsNone(workflow_runner.workflowResult())

                wait_until(lambda: self.assertTrue(workflow_runner.isRunning()))
                wait_until(lambda: self.assertFileExists("wait_started_0"))

                wait_until(lambda: self.assertFileExists("wait_finished_0"))

                wait_until(lambda: self.assertFileExists("wait_started_1"))

                workflow_runner.cancel()

                wait_until(lambda: self.assertFileExists("wait_cancelled_1"))

                self.assertTrue(workflow_runner.isCancelled())

            self.assertFileDoesNotExist("wait_finished_1")
            self.assertFileDoesNotExist("wait_started_2")
            self.assertFileDoesNotExist("wait_cancelled_2")
            self.assertFileDoesNotExist("wait_finished_2")

    def test_workflow_thread_cancel_external(self):
        with TestAreaContext("python/job_queue/workflow_runner_external") as work_area:
            WorkflowCommon.createWaitJob()

            joblist = WorkflowJoblist()
            self.assertTrue(joblist.addJobFromFile("WAIT", "external_wait_job"))
            self.assertTrue("WAIT" in joblist)

            workflow = Workflow("wait_workflow", joblist)

            self.assertEqual(len(workflow), 3)

            workflow_runner = WorkflowRunner(
                workflow, ert=None, context=SubstitutionList()
            )

            self.assertFalse(workflow_runner.isRunning())

            with workflow_runner:
                wait_until(lambda: self.assertTrue(workflow_runner.isRunning()))
                wait_until(lambda: self.assertFileExists("wait_started_0"))
                wait_until(lambda: self.assertFileExists("wait_finished_0"))
                wait_until(lambda: self.assertFileExists("wait_started_1"))
                workflow_runner.cancel()
                self.assertTrue(workflow_runner.isCancelled())

            self.assertFileDoesNotExist("wait_finished_1")
            self.assertFileDoesNotExist("wait_started_2")
            self.assertFileDoesNotExist("wait_cancelled_2")
            self.assertFileDoesNotExist("wait_finished_2")

    def test_workflow_failed_job(self):
        with TestAreaContext("python/job_queue/workflow_runner_fails") as work_area:
            WorkflowCommon.createExternalDumpJob()

            joblist = WorkflowJoblist()
            self.assertTrue(joblist.addJobFromFile("DUMP", "dump_failing_job"))
            workflow = Workflow("dump_workflow", joblist)
            self.assertEqual(len(workflow), 2)

            workflow_runner = WorkflowRunner(
                workflow, ert=None, context=SubstitutionList()
            )

            self.assertFalse(workflow_runner.isRunning())
            with patch.object(
                Workflow, "run", side_effect=Exception("mocked workflow error")
            ), workflow_runner:
                workflow_runner.wait()
                self.assertNotEqual(workflow_runner.exception(), None)

    def test_workflow_success(self):
        with TestAreaContext("python/job_queue/workflow_runner_fast") as work_area:
            WorkflowCommon.createWaitJob()

            joblist = WorkflowJoblist()
            self.assertTrue(joblist.addJobFromFile("WAIT", "wait_job"))
            self.assertTrue(
                joblist.addJobFromFile("EXTERNAL_WAIT", "external_wait_job")
            )

            workflow = Workflow("fast_wait_workflow", joblist)

            self.assertEqual(len(workflow), 2)

            workflow_runner = WorkflowRunner(
                workflow, ert=None, context=SubstitutionList()
            )

            self.assertFalse(workflow_runner.isRunning())
            with workflow_runner:
                workflow_runner.wait()
            self.assertFileExists("wait_started_0")
            self.assertFileDoesNotExist("wait_cancelled_0")
            self.assertFileExists("wait_finished_0")

            self.assertFileExists("wait_started_1")
            self.assertFileDoesNotExist("wait_cancelled_1")
            self.assertFileExists("wait_finished_1")

            self.assertTrue(workflow_runner.workflowResult())
