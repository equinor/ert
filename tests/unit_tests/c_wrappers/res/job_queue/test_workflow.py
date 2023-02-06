import pytest

from ert._c_wrappers.config.config_parser import ConfigValidationError
from ert._c_wrappers.job_queue import Workflow, WorkflowJob
from ert._c_wrappers.util.substitution_list import SubstitutionList

from .workflow_common import WorkflowCommon


@pytest.mark.usefixtures("use_tmpdir")
def test_workflow():
    WorkflowCommon.createExternalDumpJob()

    dump_job = WorkflowJob.fromFile("dump_job", name="DUMP")

    with pytest.raises(ConfigValidationError, match="Could not open config_file"):
        _ = WorkflowJob.fromFile("knock_job", name="KNOCK")

    workflow = Workflow.from_file("dump_workflow", None, {"DUMP": dump_job})

    assert len(workflow) == 2

    job, args = workflow[0]
    assert args[0] == "dump1"
    assert args[1] == "dump_text_1"

    job, args = workflow[1]
    assert job.name == "DUMP"


@pytest.mark.usefixtures("use_tmpdir")
def test_workflow_run():
    WorkflowCommon.createExternalDumpJob()

    dump_job = WorkflowJob.fromFile("dump_job", name="DUMP")

    context = SubstitutionList()
    context.addItem("<PARAM>", "text")

    workflow = Workflow.from_file("dump_workflow", context, {"DUMP": dump_job})

    assert len(workflow) == 2

    assert workflow.run(None)

    with open("dump1", "r", encoding="utf-8") as f:
        assert f.read() == "dump_text_1"

    with open("dump2", "r", encoding="utf-8") as f:
        assert f.read() == "dump_text_2"


@pytest.mark.usefixtures("use_tmpdir")
def test_failing_workflow_run():
    WorkflowCommon.createExternalDumpJob()
    with pytest.raises(ValueError, match="does not exist"):
        _ = Workflow.from_file("undefined", None, {})
