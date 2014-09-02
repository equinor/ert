import ctypes
from ert.cwrap import clib, CWrapper
from ert.test import ExtendedTestCase, TestAreaContext
from ert_tests.job_queue.workflow_common import WorkflowCommon

test_lib  = clib.ert_load("libjob_queue") # create a local namespace
cwrapper =  CWrapper(test_lib)

alloc_config = cwrapper.prototype("c_void_p workflow_job_alloc_config()")
alloc_from_file = cwrapper.prototype("workflow_job_obj workflow_job_config_alloc(char*, c_void_p, char*)")

class FunctionErtScriptTest(ExtendedTestCase):

    def test_power(self):
        with TestAreaContext("python/job_queue/workflow_job") as work_area:
            WorkflowCommon.createInternalFunctionJob()

            config = alloc_config()
            workflow_job = alloc_from_file("CONCAT", config, "concatenate_job")

            result = workflow_job.run(None, ["Str", "ing"])
            result = ctypes.c_char_p(result) # return type is a (void*) pointer -> convert to char*
            self.assertEqual(result.value, "String")

