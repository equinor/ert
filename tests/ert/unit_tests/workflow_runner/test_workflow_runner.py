import os.path
from unittest.mock import patch

import pytest

from ert import WorkflowRunner
from ert.config import Workflow, WorkflowJob
from ert.substitutions import Substitutions
from tests.ert.utils import wait_until

from .workflow_common import WorkflowCommon


@pytest.mark.integration_test
@pytest.mark.usefixtures("use_tmpdir")
def test_workflow_thread_cancel_ert_script():
    WorkflowCommon.createWaitJob()

    wait_job = WorkflowJob.from_file("wait_job", name="WAIT")

    workflow = Workflow.from_file("wait_workflow", Substitutions(), {"WAIT": wait_job})

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
    workflow = Workflow.from_file("wait_workflow", Substitutions(), {"WAIT": wait_job})

    assert len(workflow) == 3

    workflow_runner = WorkflowRunner(workflow)

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
    workflow = Workflow.from_file("dump_workflow", Substitutions(), {"DUMP": dump_job})
    assert len(workflow) == 2

    workflow_runner = WorkflowRunner(workflow)

    assert not workflow_runner.isRunning()
    with (
        patch.object(
            WorkflowRunner,
            "run_blocking",
            side_effect=Exception("mocked workflow error"),
        ),
        workflow_runner,
    ):
        workflow_runner.wait()
        assert workflow_runner.exception() is not None


@pytest.mark.integration_test
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
        Substitutions(),
        {"WAIT": wait_job, "EXTERNAL_WAIT": external_job},
    )

    assert len(workflow) == 2

    workflow_runner = WorkflowRunner(workflow)

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


@pytest.mark.usefixtures("use_tmpdir")
def test_workflow_stops_with_stopping_job():
    WorkflowCommon.createExternalDumpJob()
    with open("dump_failing_job", "a", encoding="utf-8") as f:
        f.write("STOP_ON_FAIL True")

    with open("dump_failing_workflow", "w", encoding="utf-8") as f:
        f.write("DUMP")

    job_failing_dump = WorkflowJob.from_file("dump_failing_job")
    assert job_failing_dump.stop_on_fail

    workflow = Workflow.from_file(
        src_file="dump_failing_workflow",
        context=Substitutions(),
        job_dict={"DUMP": job_failing_dump},
    )

    runner = WorkflowRunner(workflow)
    with pytest.raises(RuntimeError, match="Workflow job dump_failing_job failed"):
        runner.run_blocking()

    with open("dump_failing_job", "a", encoding="utf-8") as f:
        f.write("\nSTOP_ON_FAIL False")

    job_successful_dump = WorkflowJob.from_file("dump_failing_job")
    assert not job_successful_dump.stop_on_fail
    workflow = Workflow.from_file(
        src_file="dump_failing_workflow",
        context=Substitutions(),
        job_dict={"DUMP": job_successful_dump},
    )

    # Expect no error raised
    WorkflowRunner(workflow).run_blocking()
