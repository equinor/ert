from ert.cwrap import clib, CWrapper
from ert.job_queue import WorkflowJob
from ert.job_queue.workflow_job_monitor import WorkflowJobMonitor
from ert.test import TestAreaContext
from ert.test.extended_testcase import ExtendedTestCase
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


    def createWorkflowJobs(self):
        with open("workflow_job_1", "w") as f:
            f.write("INTERNAL True\n")
            f.write("FUNCTION enkf_main_select_case_JOB\n")
            f.write("MIN_ARG 1\n")
            f.write("MAX_ARG 1\n")
            f.write("ARG_TYPE 0 STRING\n")

        with open("workflow_job_2", "w") as f:
            f.write("INTERNAL True\n")
            f.write("SCRIPT /path/to/script\n")
            f.write("MIN_ARG 1\n")
            f.write("MAX_ARG 1\n")
            f.write("ARG_TYPE 0 STRING\n")


        with open("workflow_job_args", "w") as f:
            f.write("INTERNAL True\n")
            f.write("SCRIPT /path/to/arg_script\n")
            f.write("MIN_ARG 4\n")
            f.write("MAX_ARG 5\n")
            f.write("ARG_TYPE 0 STRING\n")
            f.write("ARG_TYPE 1 INT\n")
            f.write("ARG_TYPE 2 FLOAT\n")
            f.write("ARG_TYPE 3 BOOL\n")
            f.write("ARG_TYPE 4 STRING\n")



    def test_read_internal_function(self):
        with TestAreaContext("python/job_queue/workflow_job") as work_area:
            self.createWorkflowJobs()

            config = alloc_config()
            workflow_job = alloc_from_file("Test", config, "workflow_job_1")

            self.assertEqual(workflow_job.name(), "Test")
            self.assertTrue(workflow_job.isInternal())

            self.assertFalse(workflow_job.isInternalScript())
            self.assertIsNone(workflow_job.getInternalScriptPath())


            workflow_job = alloc_from_file("TestScript", config, "workflow_job_2")
            self.assertEqual(workflow_job.name(), "TestScript")
            self.assertTrue(workflow_job.isInternal())

            self.assertTrue(workflow_job.isInternalScript())
            self.assertEqual(workflow_job.getInternalScriptPath(), "/path/to/script")


    def test_read_config_file(self):
        with TestAreaContext("python/job_queue/workflow_job") as work_area:
            self.createWorkflowJobs()

            config = alloc_config()
            workflow_job = alloc_from_file("Test", config, "workflow_job_1")

            self.assertEqual(workflow_job.name(), "Test")
            self.assertTrue(workflow_job.isInternal())

            self.assertFalse(workflow_job.isInternalScript())
            self.assertIsNone(workflow_job.getInternalScriptPath())


            workflow_job = alloc_from_file("TestScript", config, "workflow_job_2")
            self.assertEqual(workflow_job.name(), "TestScript")
            self.assertTrue(workflow_job.isInternal())

            self.assertTrue(workflow_job.isInternalScript())
            self.assertEqual(workflow_job.getInternalScriptPath(), "/path/to/script")


    def test_arguments(self):
        with TestAreaContext("python/job_queue/workflow_job") as work_area:
            self.createWorkflowJobs()

            config = alloc_config()
            job = alloc_from_file("ArgsTest", config, "workflow_job_args")

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


