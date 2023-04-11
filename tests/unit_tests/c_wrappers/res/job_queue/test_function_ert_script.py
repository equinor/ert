import pytest

from ert._c_wrappers.job_queue import WorkflowJob
from ert.parsing import ConfigValidationError

from .workflow_common import WorkflowCommon


@pytest.mark.usefixtures("use_tmpdir")
def test_compare():
    WorkflowCommon.createInternalFunctionJob()

    with pytest.raises(ConfigValidationError):
        WorkflowJob.fromFile("no/such/file")
