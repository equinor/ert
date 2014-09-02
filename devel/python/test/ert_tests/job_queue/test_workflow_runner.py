import os
import time
from ert.job_queue import WorkflowJoblist, Workflow, WorkflowRunner
from ert.test import TestAreaContext, ExtendedTestCase
from ert.util.substitution_list import SubstitutionList
from ert_tests.job_queue.workflow_common import WorkflowCommon


class WorkflowRunnerTest(ExtendedTestCase):

    def test_workflow_thread_cancel_ert_script(self):
        with TestAreaContext("python/job_queue/workflow_runner_ert_script") as work_area:
            WorkflowCommon.createWaitJob()

            joblist = WorkflowJoblist()
            self.assertTrue(joblist.addJobFromFile("WAIT", "wait_job"))
            self.assertTrue("WAIT" in joblist)

            workflow = Workflow("wait_workflow", joblist)

            self.assertEqual(len(workflow), 1)


            context = SubstitutionList()

            workflow_runner = WorkflowRunner(workflow)

            self.assertFalse(workflow_runner.isRunning())

            workflow_runner.run(None, context)

            time.sleep(1) # wait for workflow to start
            self.assertTrue(workflow_runner.isRunning())
            self.assertTrue(os.path.exists("wait_started"))

            workflow_runner.cancel()
            time.sleep(1)
            self.assertTrue(os.path.exists("wait_cancelled"))
            self.assertFalse(os.path.exists("wait_finished"))

            self.assertTrue(workflow_runner.isCancelled())

            workflow_runner.wait()



    def test_workflow_thread_cancel_external(self):
        with TestAreaContext("python/job_queue/workflow_runner_external") as work_area:
            WorkflowCommon.createWaitJob()

            joblist = WorkflowJoblist()
            self.assertTrue(joblist.addJobFromFile("WAIT", "external_wait_job"))
            self.assertTrue("WAIT" in joblist)

            workflow = Workflow("wait_workflow", joblist)

            self.assertEqual(len(workflow), 1)


            context = SubstitutionList()

            workflow_runner = WorkflowRunner(workflow)

            self.assertFalse(workflow_runner.isRunning())

            workflow_runner.run(None, context)

            time.sleep(1) # wait for workflow to start
            self.assertTrue(workflow_runner.isRunning())
            self.assertTrue(os.path.exists("wait_started"))

            workflow_runner.cancel()
            time.sleep(1)
            # Cancel file not created since the python script is run as external...
            self.assertFalse(os.path.exists("wait_finished"))

            self.assertTrue(workflow_runner.isCancelled())

            workflow_runner.wait()
