import os.path
from unittest.mock import patch

import pytest

from ert._c_wrappers.util.substitution_list import SubstitutionList
from ert.config import Workflow, WorkflowJob
from ert.job_queue import WorkflowRunner
from tests.utils import wait_until

from .workflow_common import WorkflowCommon


@pytest.mark.usefixtures("use_tmpdir")
def test_workflow_thread_cancel_ert_script():
    WorkflowCommon.createWaitJob()

    wait_job = WorkflowJob.from_file("wait_job", name="WAIT")

    workflow = Workflow.from_file(
        "wait_workflow", SubstitutionList(), {"WAIT": wait_job}
    )

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


@pytest.mark.usefixtures("use_tmpdir")
def test_workflow_thread_cancel_external():
    WorkflowCommon.createWaitJob()

    wait_job = WorkflowJob.from_file(
        name="WAIT",
        config_file="wait_job",
    )
    workflow = Workflow.from_file(
        "wait_workflow", SubstitutionList(), {"WAIT": wait_job}
    )

    assert len(workflow) == 3

    workflow_runner = WorkflowRunner(workflow, ert=None)

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


@pytest.mark.usefixtures("use_tmpdir")
def test_workflow_failed_job():
    WorkflowCommon.createExternalDumpJob()

    dump_job = WorkflowJob.from_file(
        name="DUMP",
        config_file="dump_job",
    )
    workflow = Workflow.from_file(
        "dump_workflow", SubstitutionList(), {"DUMP": dump_job}
    )
    assert len(workflow) == 2

    workflow_runner = WorkflowRunner(workflow, ert=None)

    assert not workflow_runner.isRunning()
    with patch.object(
        WorkflowRunner, "run_blocking", side_effect=Exception("mocked workflow error")
    ), workflow_runner:
        workflow_runner.wait()
        assert workflow_runner.exception() is not None


@pytest.mark.usefixtures("use_tmpdir")
def test_workflow_success():
    WorkflowCommon.createWaitJob()

    external_job = WorkflowJob.from_file(
        name="EXTERNAL_WAIT", config_file="external_wait_job"
    )
    wait_job = WorkflowJob.from_file(
        name="WAIT",
        config_file="wait_job",
    )
    workflow = Workflow.from_file(
        "fast_wait_workflow",
        SubstitutionList(),
        {"WAIT": wait_job, "EXTERNAL_WAIT": external_job},
    )

    assert len(workflow) == 2

    workflow_runner = WorkflowRunner(workflow, ert=None)

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
