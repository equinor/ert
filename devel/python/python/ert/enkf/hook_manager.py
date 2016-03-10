import os
import sys
from ert.cwrap import BaseCClass, CWrapper
from ert.enkf import ENKF_LIB


class HookManager(BaseCClass):

    def __init__(self):
        raise NotImplementedError("Class can not be instantiated directly!")


    def checkRunpathListFile(self):
        """ @rtype: bool """
        runpath_list_file = HookManager.cNamespace().get_runpath_list_file(self)

        if not os.path.exists(runpath_list_file):
            sys.stderr.write("** Warning: the file: %s with a list of runpath directories was not found - hook workflow will probably fail.\n" % runpath_list_file)
    
    def getRunpathList(self):
        """ @rtype: RunpathList """
        return HookManager.cNamespace().get_runpath_list(self)
    
    
    def runWorkflows(self , run_time , ert_self):
        ert_self_ptr = ert_self.from_param(ert_self).value
        HookManager.cNamespace().run_workflows(self , run_time , ert_self_ptr)
        
    
cwrapper = CWrapper(ENKF_LIB)

cwrapper.registerObjectType("hook_manager", HookManager)

HookManager.cNamespace().get_runpath_list_file = cwrapper.prototype("char* hook_manager_get_runpath_list_file(hook_manager)")
HookManager.cNamespace().run_workflows = cwrapper.prototype("void hook_manager_run_workflows(hook_manager , hook_runtime_enum , void*)")
