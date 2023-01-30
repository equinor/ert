import pytest

from ert._c_wrappers.job_queue import WorkflowJob

from .workflow_common import WorkflowCommon


@pytest.mark.usefixtures("use_tmpdir")
def test_read_internal_function():
    WorkflowCommon.createInternalFunctionJob()
    WorkflowCommon.createErtScriptsJob()

    workflow_job = WorkflowJob.fromFile(
        name="SUBTRACT",
        config_file="subtract_script_job",
    )
    assert workflow_job.name == "SUBTRACT"
    assert workflow_job.internal
    assert workflow_job.function is None

    assert workflow_job.script.endswith("subtract_script.py")


@pytest.mark.usefixtures("use_tmpdir")
def test_arguments():
    WorkflowCommon.createInternalFunctionJob()
    WorkflowCommon.createErtScriptsJob()

    job = WorkflowJob.fromFile(
        name="SUBTRACT",
        config_file="subtract_script_job",
    )

    assert job.min_args == 2
    assert job.max_args == 2
    assert job.argumentTypes() == [float, float]

    job.run(None, [1, 2.5])

    with pytest.raises(ValueError, match="requires at least 2 arguments"):
        job.run(None, [1])

    with pytest.raises(ValueError, match="can only have 2 arguments"):
        job.run(None, ["x %d %f %d %s", 1, 2.5, True, "y", "nada"])


@pytest.mark.usefixtures("use_tmpdir")
def test_run_external_job():
    WorkflowCommon.createExternalDumpJob()

    job = WorkflowJob.fromFile(
        name="DUMP",
        config_file="dump_job",
    )

    assert not job.internal
    argTypes = job.argumentTypes()
    job.run(None, ["test", "text"])
    assert argTypes == [str, str]

    assert job.run_status.stdoutdata == "Hello World\n"

    with open("test", "r", encoding="utf-8") as f:
        assert f.read() == "text"


@pytest.mark.usefixtures("use_tmpdir")
def test_error_handling_external_job():
    WorkflowCommon.createExternalDumpJob()

    job = WorkflowJob.fromFile(
        name="DUMP",
        config_file="dump_failing_job",
    )

    assert not job.internal
    job.argumentTypes()
    job.run(None, [])
    job_status = job.run_status
    assert job_status.has_failed()
    assert "Traceback" in job_status.stderrdata


@pytest.mark.usefixtures("use_tmpdir")
def test_run_internal_script():
    WorkflowCommon.createErtScriptsJob()

    job = WorkflowJob.fromFile(
        name="SUBTRACT",
        config_file="subtract_script_job",
    )

    job.run(None, ["1", "2"])
    job_status = job.run_status

    assert job_status.has_finished()
    assert job_status.stdoutdata == -1
