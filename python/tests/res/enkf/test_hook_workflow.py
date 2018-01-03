from res.enkf.enums import HookRuntime
from tests import ResTest


class HookWorkFlowTest(ResTest):

    def test_enum(self):
        self.assertEnumIsFullyDefined(HookRuntime, "hook_run_mode_enum" , "libenkf/include/ert/enkf/hook_workflow.h", verbose=True)

