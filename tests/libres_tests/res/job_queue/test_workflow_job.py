import pytest

from ert._c_wrappers.job_queue import WorkflowJob

from .workflow_common import WorkflowCommon


@pytest.mark.usefixtures("setup_tmpdir")
def test_workflow_job_creation():
    workflow_job = WorkflowJob("Test")

    assert workflow_job.isInternal()
    assert workflow_job.name() == "Test"


@pytest.mark.usefixtures("setup_tmpdir")
def test_read_internal_function():
    WorkflowCommon.createInternalFunctionJob()
    WorkflowCommon.createErtScriptsJob()

    config = WorkflowJob.configParser()

    workflow_job = WorkflowJob.fromFile(
        name="SUBTRACT",
        parser=config,
        config_file="subtract_script_job",
    )
    assert workflow_job.name() == "SUBTRACT"
    assert workflow_job.isInternal()
    assert workflow_job.functionName() is None

    assert workflow_job.isInternalScript()
    assert workflow_job.getInternalScriptPath().endswith("subtract_script.py")


@pytest.mark.usefixtures("setup_tmpdir")
def test_arguments():
    WorkflowCommon.createInternalFunctionJob()
    WorkflowCommon.createErtScriptsJob()

    config = WorkflowJob.configParser()

    job = WorkflowJob.fromFile(
        name="SUBTRACT",
        parser=config,
        config_file="subtract_script_job",
    )

    assert job.minimumArgumentCount() == 2
    assert job.maximumArgumentCount() == 2
    assert job.argumentTypes() == [float, float]

    assert job.run(None, [1, 2.5])

    with pytest.raises(UserWarning):  # Too few arguments
        job.run(None, [1])

    with pytest.raises(UserWarning):  # Too many arguments
        job.run(None, ["x %d %f %d %s", 1, 2.5, True, "y", "nada"])


@pytest.mark.usefixtures("setup_tmpdir")
def test_run_external_job():

    WorkflowCommon.createExternalDumpJob()

    config = WorkflowJob.configParser()

    job = WorkflowJob.fromFile(
        name="DUMP",
        parser=config,
        config_file="dump_job",
    )

    assert not job.isInternal()
    argTypes = job.argumentTypes()
    assert argTypes == [str, str]
    assert job.run(None, ["test", "text"]) is None
    assert job.stdoutdata() == "Hello World\n"

    with open("test", "r") as f:
        assert f.read() == "text"


@pytest.mark.usefixtures("setup_tmpdir")
def test_error_handling_external_job():

    WorkflowCommon.createExternalDumpJob()

    config = WorkflowJob.configParser()

    job = WorkflowJob.fromFile(
        name="DUMP",
        parser=config,
        config_file="dump_failing_job",
    )

    assert not job.isInternal()
    job.argumentTypes()
    assert job.run(None, []) is None
    assert job.stderrdata().startswith("Traceback")


@pytest.mark.usefixtures("setup_tmpdir")
def test_run_internal_script():
    WorkflowCommon.createErtScriptsJob()

    config = WorkflowJob.configParser()

    job = WorkflowJob.fromFile(
        name="SUBTRACT",
        parser=config,
        config_file="subtract_script_job",
    )

    result = job.run(None, ["1", "2"])

    assert result == -1
