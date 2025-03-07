import os.path
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

import ert
from ert.config import ConfigWarning, Workflow, WorkflowJob
from ert.substitutions import Substitutions
from ert.workflow_runner import WorkflowJobRunner, WorkflowRunner
from tests.ert.utils import wait_until

from .workflow_common import WorkflowCommon


@pytest.mark.usefixtures("use_tmpdir")
def test_read_internal_function():
    WorkflowCommon.createErtScriptsJob()
    with (
        pytest.warns(ConfigWarning, match="Deprecated keywords, SCRIPT and INTERNAL"),
    ):
        workflow_job = WorkflowJob.from_file(
            name="SUBTRACT",
            config_file="subtract_script_job",
        )
    assert workflow_job.name == "SUBTRACT"
    assert workflow_job.ert_script.__name__ == "SubtractScript"


@pytest.mark.usefixtures("use_tmpdir")
def test_arguments():
    WorkflowCommon.createErtScriptsJob()

    job = WorkflowJob.from_file(
        name="SUBTRACT",
        config_file="subtract_script_job",
    )

    assert job.min_args == 2
    assert job.max_args == 2
    assert job.argument_types() == [float, float]

    assert WorkflowJobRunner(job).run([1, 2.5])

    with pytest.raises(ValueError, match="requires at least 2 arguments"):
        WorkflowJobRunner(job).run([1])

    with pytest.raises(ValueError, match="can only have 2 arguments"):
        WorkflowJobRunner(job).run(["x %d %f %d %s", 1, 2.5, True, "y", "nada"])


@pytest.mark.usefixtures("use_tmpdir")
def test_run_external_job():
    WorkflowCommon.createExternalDumpJob()

    job = WorkflowJob.from_file(
        name="DUMP",
        config_file="dump_job",
    )

    assert not job.ert_script
    argTypes = job.argument_types()
    assert argTypes == [str, str]
    runner = WorkflowJobRunner(job)
    assert runner.run(["test", "text"]) is None
    assert runner.stdoutdata() == "Hello World\n"

    with open("test", encoding="utf-8") as f:
        assert f.read() == "text"


@pytest.mark.usefixtures("use_tmpdir")
def test_error_handling_external_job():
    WorkflowCommon.createExternalDumpJob()

    job = WorkflowJob.from_file(
        name="DUMP",
        config_file="dump_failing_job",
    )

    assert not job.ert_script
    job.argument_types()
    runner = WorkflowJobRunner(job)
    assert runner.run([]) is None
    assert runner.stderrdata().startswith("Traceback")


@pytest.mark.usefixtures("use_tmpdir")
def test_run_internal_script():
    WorkflowCommon.createErtScriptsJob()

    job = WorkflowJob.from_file(
        name="SUBTRACT",
        config_file="subtract_script_job",
    )

    result = WorkflowJobRunner(job).run(["1", "2"])

    assert result == -1


@pytest.mark.usefixtures("use_tmpdir")
@pytest.mark.parametrize(
    "config, expected_result",
    [
        (["INTERNAL FALSE"], "FALSE has no effect"),
        (["SCRIPT scripy.py"], "SCRIPT has no effect"),
        (["SCRIPT scripy.py", "INTERNAL TRUE"], "SCRIPT and INTERNAL"),
    ],
)
def test_deprecated_keywords(config, expected_result, monkeypatch):
    monkeypatch.setattr(ert.config.workflow_job, "ErtScript", MagicMock())
    monkeypatch.setattr(
        ert.config.workflow_job.WorkflowJob, "__post_init__", MagicMock()
    )
    with open("test_job", "w", encoding="utf-8") as f:
        f.write("\n".join(config))
    Path("script.py").touch()
    with pytest.warns(ConfigWarning, match=expected_result):
        WorkflowJob.from_file(
            name="TEST",
            config_file="test_job",
        )


@pytest.mark.usefixtures("use_tmpdir")
def test_stop_on_fail_is_parsed_internal():
    with open("fail_job", "w+", encoding="utf-8") as f:
        f.write("INTERNAL True\n")
        f.write("SCRIPT fail_script.py\n")
        f.write("MIN_ARG 1\n")
        f.write("MAX_ARG 1\n")
        f.write("ARG_TYPE 0 STRING\n")
        f.write("STOP_ON_FAIL True\n")

    with open("fail_script.py", "w+", encoding="utf-8") as f:
        f.write(
            """
from ert import ErtScript

class SevereErtFailureScript(ErtScript):
    def __init__(self, ert, storage, ensemble=None):
        assert False, "Severe ert failure"

    def run(self, *args):
        pass
            """
        )

    job_internal = WorkflowJob.from_file(
        name="FAIL",
        config_file="fail_job",
    )

    assert job_internal.stop_on_fail


@pytest.mark.usefixtures("use_tmpdir")
def test_workflow_run():
    WorkflowCommon.createExternalDumpJob()

    dump_job = WorkflowJob.from_file("dump_job", name="DUMP")

    context = Substitutions()
    context["<PARAM>"] = "text"

    workflow = Workflow.from_file("dump_workflow", context, {"DUMP": dump_job})

    assert len(workflow) == 2

    job, args = workflow[0]
    assert args[0] == "dump1"
    assert args[1] == "dump_text_1"

    job, args = workflow[1]
    assert job.name == "DUMP"

    WorkflowRunner(workflow, fixtures={}).run_blocking()

    with open("dump1", encoding="utf-8") as f:
        assert f.read() == "dump_text_1"

    with open("dump2", encoding="utf-8") as f:
        assert f.read() == "dump_text_2"


@pytest.mark.integration_test
@pytest.mark.usefixtures("use_tmpdir")
def test_workflow_thread_cancel_ert_script():
    WorkflowCommon.createWaitJob()

    wait_job = WorkflowJob.from_file("wait_job", name="WAIT")

    workflow = Workflow.from_file("wait_workflow", Substitutions(), {"WAIT": wait_job})

    assert len(workflow) == 3

    workflow_runner = WorkflowRunner(workflow, fixtures={})

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


@pytest.mark.integration_test
@pytest.mark.usefixtures("use_tmpdir")
def test_workflow_thread_cancel_external():
    WorkflowCommon.createWaitJob()

    wait_job = WorkflowJob.from_file(
        name="WAIT",
        config_file="wait_job",
    )
    workflow = Workflow.from_file("wait_workflow", Substitutions(), {"WAIT": wait_job})

    assert len(workflow) == 3

    workflow_runner = WorkflowRunner(workflow, fixtures={})

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

    workflow_runner = WorkflowRunner(workflow, fixtures={})

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

    workflow_runner = WorkflowRunner(workflow, fixtures={})

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

    runner = WorkflowRunner(workflow, fixtures={})
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
    WorkflowRunner(workflow, fixtures={}).run_blocking()
