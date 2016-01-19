import os
import sys
from ert.cwrap import BaseCClass, CWrapper
from ert.enkf import ENKF_LIB, RunpathList


class HookWorkflow(BaseCClass):

    def __init__(self):
        raise NotImplementedError("Class can not be instantiated directly!")

    def hasWorkflow(self):
        """ @rtype: bool """
        return HookWorkflow.cNamespace().has_workflow(self)
    
    def getWorkflow(self):
        """ @rtype: Workflow """
        return HookWorkflow.cNamespace().get_workflow(self)
    
    def isPreSimulation(self):
        """ @rtype: bool """
        return HookWorkflow.cNamespace().is_presimulation(self)
    
    def isPostSimulation(self):
        """ @rtype: bool """
        return HookWorkflow.cNamespace().is_postsimulation(self)


cwrapper = CWrapper(ENKF_LIB)

cwrapper.registerObjectType("hook_workflow", HookWorkflow)

HookWorkflow.cNamespace().has_workflow = cwrapper.prototype("bool hook_workflow_has_workflow(hook_workflow)")
HookWorkflow.cNamespace().get_workflow = cwrapper.prototype("workflow_ref hook_workflow_get_workflow(hook_workflow)")
HookWorkflow.cNamespace().is_presimulation = cwrapper.prototype("bool hook_workflow_is_presimulation(hook_workflow)")
HookWorkflow.cNamespace().is_postsimulation = cwrapper.prototype("bool hook_workflow_is_postsimulation(hook_workflow)")
