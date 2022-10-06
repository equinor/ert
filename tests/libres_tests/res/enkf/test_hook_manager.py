import os
import os.path

import pytest

from ert._c_wrappers.enkf import ConfigKeys, HookManager, HookRuntime, ResConfig


def without_key(a_dict, key):
    return {index: value for index, value in a_dict.items() if index != key}


@pytest.fixture
def workflows_config_dict(copy_case):
    copy_case("config/workflows")
    # create empty runpath file
    with open("runpath", "a", encoding="utf-8"):
        pass
    config_dict = {
        ConfigKeys.RUNPATH_FILE: "runpath",
        ConfigKeys.CONFIG_DIRECTORY: ".",
        ConfigKeys.CONFIG_FILE_KEY: "config",
        ConfigKeys.HOOK_WORKFLOW_KEY: [("MAGIC_PRINT", "PRE_SIMULATION")],
        # these two entries makes the workflow_list load this workflow, but
        # are not needed by hook_manager directly
        ConfigKeys.LOAD_WORKFLOW_JOB: "workflowjobs/MAGIC_PRINT",
        ConfigKeys.LOAD_WORKFLOW: "workflows/MAGIC_PRINT",
    }
    with open("config", "w+") as config:
        # necessary in the file, but irrelevant to this test
        config.write("JOBNAME  Job%d\n")
        config.write("NUM_REALIZATIONS  1\n")
        for key, val in config_dict.items():
            if key in (ConfigKeys.CONFIG_FILE_KEY, ConfigKeys.CONFIG_DIRECTORY):
                continue
            if isinstance(val, str):
                config.write(f"{key} {val}\n")
            else:
                # assume this is the list of tuple for hook workflows
                for val1, val2 in val:
                    config.write(f"{key} {val1} {val2}\n")
    return config_dict


def test_different_hook_workflow_gives_not_equal_hook_managers(workflows_config_dict):
    res_config = ResConfig(
        user_config_file=workflows_config_dict[ConfigKeys.CONFIG_FILE_KEY]
    )
    assert HookManager(
        workflow_list=res_config.ert_workflow_list,
        config_dict=workflows_config_dict,
    ) != HookManager(
        workflow_list=res_config.ert_workflow_list,
        config_dict=without_key(workflows_config_dict, ConfigKeys.HOOK_WORKFLOW_KEY),
    )


def test_old_and_new_constructor_creates_equal_config(workflows_config_dict):
    res_config = ResConfig(
        user_config_file=workflows_config_dict[ConfigKeys.CONFIG_FILE_KEY]
    )
    assert res_config.hook_manager == HookManager(
        workflow_list=res_config.ert_workflow_list,
        config_dict=workflows_config_dict,
    )


def test_all_config_entries_are_set(workflows_config_dict):
    res_config = ResConfig(
        user_config_file=workflows_config_dict[ConfigKeys.CONFIG_FILE_KEY]
    )
    hook_manager = HookManager(
        workflow_list=res_config.ert_workflow_list,
        config_dict=workflows_config_dict,
    )

    assert len(hook_manager) == 1

    magic_workflow = hook_manager[0]
    assert magic_workflow.getWorkflow().src_file == os.path.join(
        os.getcwd(), workflows_config_dict[ConfigKeys.LOAD_WORKFLOW]
    )
    assert magic_workflow.getRunMode() == HookRuntime.PRE_SIMULATION
