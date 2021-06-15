from utils import ResTest

from res.enkf.enums import HookRuntime


class HookWorkFlowTest(ResTest):
    def test_enum(self):
        self.assertEnumIsFullyDefined(
            HookRuntime,
            "hook_run_mode_enum",
            "lib/include/ert/enkf/hook_workflow.hpp",
            verbose=True,
        )
