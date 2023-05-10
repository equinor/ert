import pytest

from ert._c_wrappers.job_queue import WorkflowJob, WorkflowJobRunner

from .workflow_common import WorkflowCommon


@pytest.mark.usefixtures("use_tmpdir")
def test_read_internal_function():
    WorkflowCommon.createErtScriptsJob()

    workflow_job = WorkflowJob.fromFile(
        name="SUBTRACT",
        config_file="subtract_script_job",
    )
    assert workflow_job.name == "SUBTRACT"
    assert workflow_job.internal

    assert workflow_job.script.endswith("subtract_script.py")


@pytest.mark.usefixtures("use_tmpdir")
def test_arguments():
    WorkflowCommon.createErtScriptsJob()

    job = WorkflowJob.fromFile(
        name="SUBTRACT",
        config_file="subtract_script_job",
    )

    assert job.min_args == 2
    assert job.max_args == 2
    assert job.argumentTypes() == [float, float]

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

    job = WorkflowJob.fromFile(
        name="DUMP",
        config_file="dump_job",
    )

    assert not job.internal
    argTypes = job.argumentTypes()
    assert argTypes == [str, str]
    runner = WorkflowJobRunner(job)
    assert runner.run(None, None, None, ["test", "text"]) is None
    assert runner.stdoutdata() == "Hello World\n"

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
    runner = WorkflowJobRunner(job)
    assert runner.run(None, None, None, []) is None
    assert runner.stderrdata().startswith("Traceback")


@pytest.mark.usefixtures("use_tmpdir")
def test_run_internal_script():
    WorkflowCommon.createErtScriptsJob()

    job = WorkflowJob.fromFile(
        name="SUBTRACT",
        config_file="subtract_script_job",
    )

    result = WorkflowJobRunner(job).run(None, None, None, ["1", "2"])

    assert result == -1


@pytest.mark.usefixtures("use_tmpdir")
def test_superior_parser():
    WorkflowCommon.createErtScriptsJob()

    job = WorkflowJob.fromFile(
        name="SUBTRACT",
        config_file="subtract_script_job",
    )

    result = WorkflowJobRunner(job).run(None, None, None, ["1", "2"])

    assert result == -1
