from workflow_common import WorkflowCommon
from ert.test import TestAreaContext, ExtendedTestCase


from ert.job_queue.ext_job import ExtJob



class ExtJobTest(ExtendedTestCase):
    def test_load_forward_model(self):
        with TestAreaContext("python/job_queue/forward_model") as work_area:
            WorkflowCommon.createExternalDumpJob()
            try:
                ExtJob("dump_job",work_area.get_cwd(),False,"dump_job")
            except :
                self.fail("ExtJob() raised Exception unexpectedly!")


    def test_load_forward_model_with_error1(self):
        with TestAreaContext("python/job_queue/forward_model") as work_area:
            self.assertRaises(ValueError, ExtJob,"dump_job",work_area.get_cwd(),False,"test")


    def test_load_forward_model_with_error2(self):
        with TestAreaContext("python/job_queue/forward_model") as work_area:
            WorkflowCommon.createExternalDumpJobWithError()
            self.assertRaises(ValueError, ExtJob,"dump_job",work_area.get_cwd(),False,"dump_job_error")
