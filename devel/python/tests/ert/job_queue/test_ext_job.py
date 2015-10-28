from workflow_common import WorkflowCommon
from ert.test import TestAreaContext
from ert_statoil.testcase import TestCase

from ert.job_queue.ext_job import ExtJob



class ExtJobTest(TestCase):
    def test_load_forward_model(self):
        with TestAreaContext("python/job_queue/workflow_job",store_area=True) as work_area:
            WorkflowCommon.createExternalDumpJob()
            
            try:
                ExtJob("dump_job",work_area.get_cwd(),False,"dump_job")
            except :
                self.fail("ExtJob() raised Exception unexpectedly!")


    def test_load_forward_model_with_error(self):
        with TestAreaContext("python/job_queue/workflow_job",store_area=True) as work_area:

            test = work_area.get_cwd()

            self.assertRaises(ValueError, ExtJob,"dump_job",test,False,"test")


