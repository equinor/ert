from ecl.util.test import TestAreaContext
from res import ResPrototype
from res.job_queue import WorkflowJob

from ...libres_utils import ResTest
from .workflow_common import WorkflowCommon


class _TestWorkflowJobPrototype(ResPrototype):
    def __init__(self, prototype, bind=True):
        super().__init__(prototype, bind=bind)


class WorkflowJobTest(ResTest):
    _alloc_config = _TestWorkflowJobPrototype(
        "config_parser_obj workflow_job_alloc_config()", bind=False
    )
    _alloc_from_file = _TestWorkflowJobPrototype(
        "workflow_job_obj workflow_job_config_alloc( char* , config_parser , char*)",
        bind=False,
    )

    def test_workflow_job_creation(self):
        workflow_job = WorkflowJob("Test")

        self.assertTrue(workflow_job.isInternal())
        self.assertEqual(workflow_job.name(), "Test")

    def test_read_internal_function(self):
        with TestAreaContext("python/job_queue/workflow_job"):
            WorkflowCommon.createInternalFunctionJob()
            WorkflowCommon.createErtScriptsJob()

            config = self._alloc_config()

            workflow_job = self._alloc_from_file(
                "SUBTRACT", config, "subtract_script_job"
            )
            self.assertEqual(workflow_job.name(), "SUBTRACT")
            self.assertTrue(workflow_job.isInternal())
            self.assertIsNone(workflow_job.functionName())

            self.assertTrue(workflow_job.isInternalScript())
            self.assertTrue(
                workflow_job.getInternalScriptPath().endswith("subtract_script.py")
            )

    def test_arguments(self):
        with TestAreaContext("python/job_queue/workflow_job"):
            WorkflowCommon.createInternalFunctionJob()

            config = self._alloc_config()
            job = self._alloc_from_file("PRINTF", config, "printf_job")

            self.assertEqual(job.minimumArgumentCount(), 4)
            self.assertEqual(job.maximumArgumentCount(), 5)
            self.assertEqual(job.argumentTypes(), [str, int, float, bool, str])

            self.assertTrue(job.run(None, ["x %d %f %d", 1, 2.5, True]))
            self.assertTrue(job.run(None, ["x %d %f %d %s", 1, 2.5, True, "y"]))

            with self.assertRaises(UserWarning):  # Too few arguments
                job.run(None, ["x %d %f", 1, 2.5])

            with self.assertRaises(UserWarning):  # Too many arguments
                job.run(None, ["x %d %f %d %s", 1, 2.5, True, "y", "nada"])

    def test_run_external_job(self):

        with TestAreaContext("python/job_queue/workflow_job"):
            WorkflowCommon.createExternalDumpJob()

            config = self._alloc_config()
            job = self._alloc_from_file("DUMP", config, "dump_job")

            self.assertFalse(job.isInternal())
            argTypes = job.argumentTypes()
            self.assertEqual(argTypes, [str, str])
            self.assertIsNone(job.run(None, ["test", "text"]))
            self.assertEqual(job.stdoutdata(), "Hello World\n")

            with open("test", "r") as f:
                self.assertEqual(f.read(), "text")

    def test_error_handling_external_job(self):

        with TestAreaContext("python/job_queue/workflow_job"):
            WorkflowCommon.createExternalDumpJob()

            config = self._alloc_config()
            job = self._alloc_from_file("DUMP", config, "dump_failing_job")

            self.assertFalse(job.isInternal())
            job.argumentTypes()
            self.assertIsNone(job.run(None, []))
            self.assertTrue(job.stderrdata().startswith("Traceback"))

    def test_run_internal_script(self):
        with TestAreaContext("python/job_queue/workflow_job"):
            WorkflowCommon.createErtScriptsJob()

            config = self._alloc_config()
            job = self._alloc_from_file("SUBTRACT", config, "subtract_script_job")

            result = job.run(None, ["1", "2"])

            self.assertEqual(result, -1)
