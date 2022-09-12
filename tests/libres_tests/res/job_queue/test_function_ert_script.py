import pytest

from ert._c_wrappers.job_queue import WorkflowJob

from .workflow_common import WorkflowCommon


@pytest.mark.usefixtures("setup_tmpdir")
def test_compare():
    WorkflowCommon.createInternalFunctionJob()

    with pytest.raises(IOError):
        WorkflowJob.fromFile("no/such/file")
