import os
import sys
from cwrap import BaseCClass
from res import ResPrototype

class HookManager(BaseCClass):
    TYPE_NAME = "hook_manager"

    _alloc                 = ResPrototype("void* hook_manager_alloc(ert_workflow_list, config_content)", bind=False)
    _free                  = ResPrototype("void hook_manager_free(subst_config)")
    _get_runpath_list_file = ResPrototype("char* hook_manager_get_runpath_list_file(hook_manager)")
    _get_runpath_list      = ResPrototype("runpath_list_ref  hook_manager_get_runpath_list(hook_manager)")
    _iget_hook_workflow    = ResPrototype("hook_workflow_ref hook_manager_iget_hook_workflow(hook_manager, int)")
    _size                  = ResPrototype("int hook_manager_get_size(hook_manager)")

    def __init__(self, workflow_list, config_content):
        c_ptr = self._alloc(workflow_list, config_content)

        if c_ptr is None:
            raise ValueError('Failed to construct RNGConfig instance')

        super(HookManager, self).__init__(c_ptr)

    def __len__(self):
        """ @rtype: int """
        return self._size()

    def __repr__(self):
        return 'HookManager(size = %d) %s' % (len(self), self._ad_str())

    def __getitem__(self, index):
        """ @rtype: Hook workflow """
        assert isinstance(index, int)
        if index < 0:
            index += len(self)
        if 0 <= index < len(self):
            return self._iget_hook_workflow(index)
        else:
            raise IndexError("Invalid index.  Valid range: [0, %d)." % len(self))

    def checkRunpathListFile(self):
        """ @rtype: bool """
        runpath_list_file = self._get_runpath_list_file()

        if not os.path.exists(runpath_list_file):
            sys.stderr.write("** Warning: the file: %s with a list of runpath directories was not found - hook workflow will probably fail.\n" % runpath_list_file)

    def getRunpathList(self):
        """ @rtype: RunpathList """
        return self._get_runpath_list()

    def runWorkflows(self , run_time , ert_self):

        workflow_list = ert_self.getWorkflowList()
        for hook_workflow in self:

            if (hook_workflow.getRunMode() is not run_time):
                continue

            workflow = hook_workflow.getWorkflow()
            workflow.run(ert_self, context=workflow_list.getContext())

    def free(self):
        self._free()
