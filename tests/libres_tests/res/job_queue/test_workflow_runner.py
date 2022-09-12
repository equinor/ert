import os.path
from unittest.mock import patch

from ecl.util.test import TestAreaContext

from ert._c_wrappers.job_queue import Workflow, WorkflowJoblist, WorkflowRunner
from ert._c_wrappers.util.substitution_list import SubstitutionList
from tests.utils import wait_until

from .workflow_common import WorkflowCommon


def test_workflow_thread_cancel_ert_script():
    with TestAreaContext("python/job_queue/workflow_runner_ert_script"):
        WorkflowCommon.createWaitJob()

        joblist = WorkflowJoblist()
        assert joblist.addJobFromFile("WAIT", "wait_job")
        assert "WAIT" in joblist

        workflow = Workflow("wait_workflow", joblist)

        assert len(workflow) == 3

        workflow_runner = WorkflowRunner(workflow)

        assert not workflow_runner.isRunning()

        with workflow_runner:
            assert workflow_runner.workflowResult() is None

            wait_until(workflow_runner.isRunning)
            wait_until(lambda: os.path.exists("wait_started_0"))

            wait_until(lambda: os.path.exists("wait_finished_0"))

            wait_until(lambda: os.path.exists("wait_started_1"))

            workflow_runner.cancel()

            wait_until(lambda: os.path.exists("wait_cancelled_1"))

            assert workflow_runner.isCancelled()

        assert not os.path.exists("wait_finished_1")
        assert not os.path.exists("wait_started_2")
        assert not os.path.exists("wait_cancelled_2")
        assert not os.path.exists("wait_finished_2")


def test_workflow_thread_cancel_external():
    with TestAreaContext("python/job_queue/workflow_runner_external"):
        WorkflowCommon.createWaitJob()

        joblist = WorkflowJoblist()
        assert joblist.addJobFromFile("WAIT", "external_wait_job")
        assert "WAIT" in joblist

        workflow = Workflow("wait_workflow", joblist)

        assert len(workflow) == 3

        workflow_runner = WorkflowRunner(workflow, ert=None, context=SubstitutionList())

        assert not workflow_runner.isRunning()

        with workflow_runner:
            wait_until(workflow_runner.isRunning)
            wait_until(lambda: os.path.exists("wait_started_0"))
            wait_until(lambda: os.path.exists("wait_finished_0"))
            wait_until(lambda: os.path.exists("wait_started_1"))
            workflow_runner.cancel()
            assert workflow_runner.isCancelled()

        assert not os.path.exists("wait_finished_1")
        assert not os.path.exists("wait_started_2")
        assert not os.path.exists("wait_cancelled_2")
        assert not os.path.exists("wait_finished_2")


def test_workflow_failed_job():
    with TestAreaContext("python/job_queue/workflow_runner_fails"):
        WorkflowCommon.createExternalDumpJob()

        joblist = WorkflowJoblist()
        assert joblist.addJobFromFile("DUMP", "dump_failing_job")
        workflow = Workflow("dump_workflow", joblist)
        assert len(workflow) == 2

        workflow_runner = WorkflowRunner(workflow, ert=None, context=SubstitutionList())

        assert not workflow_runner.isRunning()
        with patch.object(
            Workflow, "run", side_effect=Exception("mocked workflow error")
        ), workflow_runner:
            workflow_runner.wait()
            assert workflow_runner.exception() != None


def test_workflow_success():
    with TestAreaContext("python/job_queue/workflow_runner_fast"):
        WorkflowCommon.createWaitJob()

        joblist = WorkflowJoblist()
        assert joblist.addJobFromFile("WAIT", "wait_job")
        assert joblist.addJobFromFile("EXTERNAL_WAIT", "external_wait_job")

        workflow = Workflow("fast_wait_workflow", joblist)

        assert len(workflow) == 2

        workflow_runner = WorkflowRunner(workflow, ert=None, context=SubstitutionList())

        assert not workflow_runner.isRunning()
        with workflow_runner:
            workflow_runner.wait()
        assert os.path.exists("wait_started_0")
        assert not os.path.exists("wait_cancelled_0")
        assert os.path.exists("wait_finished_0")

        assert os.path.exists("wait_started_1")
        assert not os.path.exists("wait_cancelled_1")
        assert os.path.exists("wait_finished_1")

        assert workflow_runner.workflowResult()
