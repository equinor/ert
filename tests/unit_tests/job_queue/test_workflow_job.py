import pytest

from ert.config import WorkflowJob
from ert.scheduler import WorkflowJobRunner

from .workflow_common import WorkflowCommon


@pytest.mark.usefixtures("use_tmpdir")
def test_read_internal_function():
    WorkflowCommon.createErtScriptsJob()

    workflow_job = WorkflowJob.from_file(
        name="SUBTRACT",
        config_file="subtract_script_job",
    )
    assert workflow_job.name == "SUBTRACT"
    assert workflow_job.internal

    assert workflow_job.script.endswith("subtract_script.py")


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

    assert WorkflowJobRunner(job).run(None, None, None, [1, 2.5])

    with pytest.raises(ValueError, match="requires at least 2 arguments"):
        WorkflowJobRunner(job).run(None, None, None, [1])

    with pytest.raises(ValueError, match="can only have 2 arguments"):
        WorkflowJobRunner(job).run(
            None, None, None, ["x %d %f %d %s", 1, 2.5, True, "y", "nada"]
        )


@pytest.mark.usefixtures("use_tmpdir")
def test_run_external_job():
    WorkflowCommon.createExternalDumpJob()

    job = WorkflowJob.from_file(
        name="DUMP",
        config_file="dump_job",
    )

    assert not job.internal
    argTypes = job.argument_types()
    assert argTypes == [str, str]
    runner = WorkflowJobRunner(job)
    assert runner.run(None, None, None, ["test", "text"]) is None
    assert runner.stdoutdata() == "Hello World\n"

    with open("test", "r", encoding="utf-8") as f:
        assert f.read() == "text"


@pytest.mark.usefixtures("use_tmpdir")
def test_error_handling_external_job():
    WorkflowCommon.createExternalDumpJob()

    job = WorkflowJob.from_file(
        name="DUMP",
        config_file="dump_failing_job",
    )

    assert not job.internal
    job.argument_types()
    runner = WorkflowJobRunner(job)
    assert runner.run(None, None, None, []) is None
    assert runner.stderrdata().startswith("Traceback")


@pytest.mark.usefixtures("use_tmpdir")
def test_run_internal_script():
    WorkflowCommon.createErtScriptsJob()

    job = WorkflowJob.from_file(
        name="SUBTRACT",
        config_file="subtract_script_job",
    )

    result = WorkflowJobRunner(job).run(None, None, None, ["1", "2"])

    assert result == -1


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
def test_stop_on_fail_is_parsed_external():
    with open("fail_job", "w+", encoding="utf-8") as f:
        f.write("INTERNAL False\n")
        f.write("EXECUTABLE echo\n")
        f.write("MIN_ARG 1\n")
        f.write("STOP_ON_FAIL True\n")

    job_internal = WorkflowJob.from_file(
        name="FAIL",
        config_file="fail_job",
    )

    assert job_internal.stop_on_fail
