import os
import sys
from cwrap import BaseCClass
from ert.enkf import EnkfPrototype

class HookManager(BaseCClass):
    TYPE_NAME = "hook_manager"

    _get_runpath_list_file = EnkfPrototype("char* hook_manager_get_runpath_list_file(hook_manager)")
    _iget_hook_workflow    = EnkfPrototype("hook_workflow_ref hook_manager_iget_hook_workflow(hook_manager, int)")
    _size                  = EnkfPrototype("int hook_manager_get_size(hook_manager)")

    def __init__(self):
        raise NotImplementedError("Class can not be instantiated directly!")

    def __len__(self):
        """ @rtype: int """
        return self._size()

    def __repr__(self):
        return 'HookManager(len = %d) at 0x%x' % (len(self), self._address())

    def __getitem__(self, index):
        """ @rtype: Hook workflow """
        assert isinstance(index, int)
        if index < len(self):
            return self._iget_hook_workflow(index)
        else:
            raise IndexError("Invalid index")

    def checkRunpathListFile(self):
        """ @rtype: bool """
        runpath_list_file = self._get_runpath_list_file()

        if not os.path.exists(runpath_list_file):
            sys.stderr.write("** Warning: the file: %s with a list of runpath directories was not found - hook workflow will probably fail.\n" % runpath_list_file)

    def getRunpathList(self):
        """ @rtype: RunpathList """
        return HookManager.cNamespace().get_runpath_list(self)

    def runWorkflows(self , run_time , ert_self):

        workflow_list = ert_self.getWorkflowList()
        for hook_workflow in self:

            if (hook_workflow.getRunMode() is not run_time):
                continue

            workflow = hook_workflow.getWorkflow()
            workflow.run(ert_self, context=workflow_list.getContext())
