from ecl.util.test import TestAreaContext
from ...libres_utils import ResTest
from res.job_queue import WorkflowJob

from .workflow_common import WorkflowCommon


class FunctionErtScriptTest(ResTest):
    def test_compare(self):
        with TestAreaContext("python/job_queue/workflow_job"):
            WorkflowCommon.createInternalFunctionJob()

            with self.assertRaises(IOError):
                WorkflowJob.fromFile("no/such/file")
