import pytest

from ert._c_wrappers.job_queue import WorkflowJob, WorkflowJoblist

from .workflow_common import WorkflowCommon


def test_workflow_joblist_creation():
    joblist = WorkflowJoblist()

    job = WorkflowJob("JOB1")

    joblist.addJob(job)

    assert job in joblist
    assert "JOB1" in joblist

    job_ref = joblist["JOB1"]

    assert job.name() == job_ref.name()


@pytest.mark.usefixtures("setup_tmpdir")
def test_workflow_joblist_with_files():
    WorkflowCommon.createErtScriptsJob()
    WorkflowCommon.createExternalDumpJob()
    WorkflowCommon.createInternalFunctionJob()

    joblist = WorkflowJoblist()

    joblist.addJobFromFile("DUMP_JOB", "dump_job")
    joblist.addJobFromFile("SUBTRACT_SCRIPT_JOB", "subtract_script_job")

    assert "DUMP_JOB" in joblist
    assert "SUBTRACT_SCRIPT_JOB" in joblist

    assert not (joblist["DUMP_JOB"]).isInternal()
    assert (joblist["SUBTRACT_SCRIPT_JOB"]).isInternal()
