from ert.cwrap import clib, CWrapper
from ert.job_queue import WorkflowJob, WorkflowJobMonitor
from ert.test import TestAreaContext, ExtendedTestCase
from ert.util import StringList
from ert_tests.job_queue.workflow_common import WorkflowCommon

test_lib  = clib.ert_load("libjob_queue") # create a local namespace
cwrapper =  CWrapper(test_lib)

alloc_config = cwrapper.prototype("c_void_p workflow_job_alloc_config()")
alloc_from_file = cwrapper.prototype("workflow_job_obj workflow_job_config_alloc(char*, c_void_p, char*)")

class WorkflowJobTest(ExtendedTestCase):

    def test_workflow_job_creation(self):
        workflow_job = WorkflowJob("Test")

        self.assertTrue(workflow_job.isInternal())
        self.assertEqual(workflow_job.name(), "Test")



    def test_read_internal_function(self):
        with TestAreaContext("python/job_queue/workflow_job") as work_area:
            WorkflowCommon.createInternalFunctionJob()
            WorkflowCommon.createErtScriptsJob()

            config = alloc_config()
            workflow_job = alloc_from_file("SELECT_CASE", config, "select_case_job")

            self.assertEqual(workflow_job.name(), "SELECT_CASE")
            self.assertTrue(workflow_job.isInternal())

            self.assertFalse(workflow_job.isInternalScript())
            self.assertIsNone(workflow_job.getInternalScriptPath())


            workflow_job = alloc_from_file("SUBTRACT", config, "subtract_script_job")
            self.assertEqual(workflow_job.name(), "SUBTRACT")
            self.assertTrue(workflow_job.isInternal())

            self.assertTrue(workflow_job.isInternalScript())
            self.assertTrue(workflow_job.getInternalScriptPath().endswith("subtract_script.py"))



    def test_arguments(self):
        with TestAreaContext("python/job_queue/workflow_job") as work_area:
            WorkflowCommon.createInternalFunctionJob()

            config = alloc_config()
            job = alloc_from_file("PRINTF", config, "printf_job")

            self.assertEqual(job.minimumArgumentCount(), 4)
            self.assertEqual(job.maximumArgumentCount(), 5)
            self.assertEqual(job.argumentTypes(), [str, int, float, bool ,str])


    def test_run_external_job(self):

        with TestAreaContext("python/job_queue/workflow_job") as work_area:
            WorkflowCommon.createExternalDumpJob()

            config = alloc_config()
            job = alloc_from_file("DUMP", config, "dump_job")

            monitor = WorkflowJobMonitor()

            arguments = StringList()
            arguments.append("test")
            arguments.append("text")

            self.assertIsNone(job.run(monitor, None, False, arguments))

            with open("test", "r") as f:
                self.assertEqual(f.read(), "text")


    def test_run_internal_script(self):
        with TestAreaContext("python/job_queue/workflow_job") as work_area:
            WorkflowCommon.createErtScriptsJob()

            config = alloc_config()
            job = alloc_from_file("SUBTRACT", config, "subtract_script_job")

            monitor = WorkflowJobMonitor()

            arguments = StringList()
            arguments.append("1")
            arguments.append("2")

            result = job.run(monitor, None, False, arguments)

            self.assertEqual(result, -1)
