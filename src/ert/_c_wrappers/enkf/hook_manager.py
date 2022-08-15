import ctypes

from cwrap import BaseCClass

from res import ResPrototype
from res.enkf.config_keys import ConfigKeys
from res.enkf.enums import HookRuntime
from res.enkf.hook_workflow import HookWorkflow


class HookManager(BaseCClass):
    TYPE_NAME = "hook_manager"

    _alloc = ResPrototype(
        "void* hook_manager_alloc(ert_workflow_list, config_content)", bind=False
    )
    # hook_manager_alloc_full() has char** which cwrap is missing an implementation for
    # so we let ctypes implicitly handle the two char** arguments and final int argument
    _alloc_full = ResPrototype(
        "void* hook_manager_alloc_full(ert_workflow_list)", bind=False
    )
    _free = ResPrototype("void hook_manager_free(hook_manager)")
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
                    f"HookManager needs {ConfigKeys.CONFIG_DIRECTORY} to be configured"
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
                        raise ValueError(f"Run mode {run_mode_name} not supported")
                    hook_workflow_names.append(hook_workflow_name)
                    hook_workflow_run_modes.append(run_mode_name)

            c_ptr = self._alloc_full(
                workflow_list,
                self._to_c_string_arr(hook_workflow_names),
                self._to_c_string_arr(hook_workflow_run_modes),
                len(hook_workflow_names),
            )

        else:
            c_ptr = self._alloc(workflow_list, config_content)

        if c_ptr is None:
            raise ValueError("Failed to construct RNGConfig instance")

        super().__init__(c_ptr)

    def _to_c_string_arr(self, L):
        arr = (ctypes.c_char_p * len(L))()
        arr[:] = [s.encode("utf-8") for s in L]
        return arr

    def __len__(self):
        """@rtype: int"""
        return self._size()

    def __repr__(self):
        return f'HookManager({", ".join([str(x) for x in self])})'

    def __getitem__(self, index) -> HookWorkflow:
        assert isinstance(index, int)
        if index < 0:
            index += len(self)
        if 0 <= index < len(self):
            return self._iget_hook_workflow(index)
        else:
            raise IndexError(f"Invalid index.  Valid range: [0, {len(self)}).")

    def runWorkflows(self, run_time, ert_self):

        workflow_list = ert_self.getWorkflowList()
        for hook_workflow in self:

            if hook_workflow.getRunMode() is not run_time:
                continue

            workflow = hook_workflow.getWorkflow()
            workflow.run(ert_self, context=workflow_list.getContext())

    def __eq__(self, other):
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
