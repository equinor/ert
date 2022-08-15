from cwrap import BaseCClass

from ert._c_wrappers import ResPrototype
from ert._c_wrappers.enkf.enums import HookRuntime
from ert._c_wrappers.job_queue import Workflow


class HookWorkflow(BaseCClass):
    TYPE_NAME = "hook_workflow"

    _get_workflow = ResPrototype(
        "workflow_ref hook_workflow_get_workflow(hook_workflow)"
    )
    _get_runmode = ResPrototype(
        "hook_runtime_enum hook_workflow_get_run_mode(hook_workflow)"
    )

    def __init__(self):
        raise NotImplementedError("Class can not be instantiated directly!")

    def getWorkflow(self) -> Workflow:
        """@rtype: Workflow"""
        return self._get_workflow()

    def getRunMode(self) -> HookRuntime:
        return self._get_runmode()

    def __eq__(self, other):
        return (
            self.getRunMode() == other.getRunMode()
            and self.getWorkflow().src_file == other.getWorkflow().src_file
        )

    def __ne__(self, other):
        return not self == other

    def __repr__(self):
        return f"HookWorkflow({self.getWorkflow().src_file}, {self.getRunMode()})"
