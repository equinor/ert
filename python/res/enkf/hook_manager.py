import os
import sys
import ctypes
from cwrap import BaseCClass
from res import ResPrototype
from res.enkf import ConfigKeys
from res.enkf.enums import HookRuntime


class HookManager(BaseCClass):
    TYPE_NAME = "hook_manager"

    _alloc = ResPrototype(
        "void* hook_manager_alloc(ert_workflow_list, config_content)", bind=False
    )
    # hook_manager_alloc_full() has char** which cwrap is missing an implementation for
    # so we let ctypes implicitly handle the two char** arguments and final int argument
    _alloc_full = ResPrototype(
        "void* hook_manager_alloc_full(ert_workflow_list, char*)", bind=False
    )
    _free = ResPrototype("void hook_manager_free(subst_config)")
    _get_runpath_list_file = ResPrototype(
        "char* hook_manager_get_runpath_list_file(hook_manager)"
    )
    _get_runpath_list = ResPrototype(
        "runpath_list_ref  hook_manager_get_runpath_list(hook_manager)"
    )
    _iget_hook_workflow = ResPrototype(
        "hook_workflow_ref hook_manager_iget_hook_workflow(hook_manager, int)"
    )
    _size = ResPrototype("int hook_manager_get_size(hook_manager)")

    def __init__(self, workflow_list, config_content=None, config_dict=None):
        if not ((config_content is not None) ^ (config_dict is not None)):
            raise ValueError("HookManager expects one of config_content or config dict")

        if config_dict is not None:
            config_dir = config_dict.get(ConfigKeys.CONFIG_DIRECTORY)
            if not isinstance(config_dir, str):
                raise ValueError(
                    "HookManager needs {} to be configured".format(
                        ConfigKeys.CONFIG_DIRECTORY
                    )
                )

            # RUNPATH_FILE #
            runpath_file_name = config_dict.get(
                ConfigKeys.RUNPATH_FILE, ConfigKeys.RUNPATH_LIST_FILE
            )
            runpath_file_path = os.path.normpath(
                os.path.join(config_dir, runpath_file_name)
            )

            # HOOK_WORKFLOW
            hook_workflow_names = []
            hook_workflow_run_modes = []
            if ConfigKeys.HOOK_WORKFLOW_KEY in config_dict:
                for hook_workflow_name, run_mode_name in config_dict[
                    ConfigKeys.HOOK_WORKFLOW_KEY
                ]:
                    if run_mode_name not in [
                        runtime.name for runtime in HookRuntime.enums()
                    ]:
                        raise ValueError(
                            "Run mode {} not supported".format(run_mode_name)
                        )
                    hook_workflow_names.append(hook_workflow_name)
                    hook_workflow_run_modes.append(run_mode_name)

            c_ptr = self._alloc_full(
                workflow_list,
                runpath_file_path,
                self._to_c_string_arr(hook_workflow_names),
                self._to_c_string_arr(hook_workflow_run_modes),
                len(hook_workflow_names),
            )

        else:
            c_ptr = self._alloc(workflow_list, config_content)

        if c_ptr is None:
            raise ValueError("Failed to construct RNGConfig instance")

        super(HookManager, self).__init__(c_ptr)

    def _to_c_string_arr(self, L):
        arr = (ctypes.c_char_p * len(L))()
        arr[:] = [s.encode("utf-8") for s in L]
        return arr

    def __len__(self):
        """@rtype: int"""
        return self._size()

    def __repr__(self):
        return "HookManager({})".format(", ".join([str(x) for x in self]))

    def __getitem__(self, index):
        """@rtype: Hook workflow"""
        assert isinstance(index, int)
        if index < 0:
            index += len(self)
        if 0 <= index < len(self):
            return self._iget_hook_workflow(index)
        else:
            raise IndexError("Invalid index.  Valid range: [0, %d)." % len(self))

    def getRunpathListFile(self):
        return self._get_runpath_list_file()

    def checkRunpathListFile(self):
        """@rtype: bool"""
        runpath_list_file = self._get_runpath_list_file()

        if not os.path.exists(runpath_list_file):
            sys.stderr.write(
                "** Warning: the file: %s with a list of runpath directories was not found - hook workflow will probably fail.\n"
                % runpath_list_file
            )

    def getRunpathList(self):
        """@rtype: RunpathList"""
        return self._get_runpath_list()

    def runWorkflows(self, run_time, ert_self):

        workflow_list = ert_self.getWorkflowList()
        for hook_workflow in self:

            if hook_workflow.getRunMode() is not run_time:
                continue

            workflow = hook_workflow.getWorkflow()
            workflow.run(ert_self, context=workflow_list.getContext())

    def __eq__(self, other):
        if self.getRunpathListFile() != other.getRunpathListFile():
            return False
        if len(self) != len(other):
            return False
        for val in self:
            if val not in other:
                return False
        return True

    def __ne__(self, other):
        return not self == other

    def free(self):
        self._free()
