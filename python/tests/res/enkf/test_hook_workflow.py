from res.enkf.enums import HookRuntime
from ecl.test import ExtendedTestCase


class HookWorkFlowTest(ExtendedTestCase):

    def test_enum(self):
        self.assertEnumIsFullyDefined(HookRuntime, "hook_run_mode_enum" , "libenkf/include/ert/enkf/hook_workflow.h", verbose=True)
        
