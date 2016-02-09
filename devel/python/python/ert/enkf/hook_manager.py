import os
import sys
from ert.cwrap import BaseCClass, CWrapper
from ert.enkf import ENKF_LIB


class HookManager(BaseCClass):

    def __init__(self):
        raise NotImplementedError("Class can not be instantiated directly!")


    def hasHookWorkflow(self):
        """ @rtype: bool """
        return HookManager.cNamespace().has_hook_workflow(self)

    def getHookWorkflow(self):
        """ @rtype: HookWorkflow """
        return HookManager.cNamespace().get_hook_workflow(self)

    def checkRunpathListFile(self):
        """ @rtype: bool """
        runpath_list_file = HookManager.cNamespace().get_runpath_list_file(self)

        if not os.path.exists(runpath_list_file):
            sys.stderr.write("** Warning: the file: %s with a list of runpath directories was not found - hook workflow will probably fail.\n" % runpath_list_file)
    
    def getRunpathList(self):
        """ @rtype: RunpathList """
        return HookManager.cNamespace().get_runpath_list(self)
    
    def hasPostHookWorkflow(self):
        """ @rtype: bool """
        return HookManager.cNamespace().has_post_hook_workflow(self)

    def getPostHookWorkflow(self):
        """ @rtype: HookWorkflow """
        return HookManager.cNamespace().get_post_hook_workflow(self)
    

    def runWorkflows(self , run_time , context):
        HookManager.cNamespace().run_workflows(self , run_time)
        
    
cwrapper = CWrapper(ENKF_LIB)

cwrapper.registerObjectType("hook_manager", HookManager)

HookManager.cNamespace().get_runpath_list_file = cwrapper.prototype("char* hook_manager_get_runpath_list_file(hook_manager)")
HookManager.cNamespace().has_hook_workflow = cwrapper.prototype("bool hook_manager_has_hook_workflow(hook_manager)")
HookManager.cNamespace().get_hook_workflow = cwrapper.prototype("hook_workflow_ref hook_manager_get_hook_workflow(hook_manager)")

HookManager.cNamespace().has_post_hook_workflow = cwrapper.prototype("bool hook_manager_has_post_hook_workflow(hook_manager)")
HookManager.cNamespace().get_post_hook_workflow = cwrapper.prototype("hook_workflow_ref hook_manager_get_post_hook_workflow(hook_manager)")
HookManager.cNamespace().run_workflows = cwrapper.prototype("void hook_manager_run_workflows(hook_manager , hook_runtime_enum , void*)")
