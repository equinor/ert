#  Copyright (C) 2017  Equinor ASA, Norway.
#
#  The file 'test_res_config.py' is part of ERT - Ensemble based Reservoir Tool.
#
#  ERT is free software: you can redistribute it and/or modify
#  it under the terms of the GNU General Public License as published by
#  the Free Software Foundation, either version 3 of the License, or
#  (at your option) any later version.
#
#  ERT is distributed in the hope that it will be useful, but WITHOUT ANY
#  WARRANTY; without even the implied warranty of MERCHANTABILITY or
#  FITNESS FOR A PARTICULAR PURPOSE.
#
#  See the GNU General Public License at <http://www.gnu.org/licenses/gpl.html>
#  for more details.
import os
import os.path
import shutil

from ert._c_wrappers.enkf import ConfigKeys, HookManager, HookRuntime, ResConfig


def workflow_example(tmpdir, source_root):
    with tmpdir.as_cwd():
        shutil.copytree(
            os.path.join(source_root, "test-data", "config", "workflows"), "test_data"
        )
        os.chdir("test_data")
        with open("config", "w+") as config:
            config.write("JOBNAME  Job%d\n")
            config.write("NUM_REALIZATIONS  1\n")
            config.write("RUNPATH_FILE  runpath\n")
            config.write("HOOK_WORKFLOW MAGIC_PRINT PRE_SIMULATION\n")
        open("runpath", "a").close()
        yield ResConfig("config")


def workflow_example_dict(workflow_example):
    return {
        ConfigKeys.RUNPATH_FILE: "runpath",
        ConfigKeys.CONFIG_DIRECTORY: os.getcwd(),
        ConfigKeys.CONFIG_FILE_KEY: "config",
        ConfigKeys.HOOK_WORKFLOW_KEY: [("MAGIC_PRINT", "PRE_SIMULATION")],
        # these two entries makes the workflow_list load this workflow, but
        # are not needed by hook_manager directly
        ConfigKeys.LOAD_WORKFLOW_JOB: "workflowjobs/MAGIC_PRINT",
        ConfigKeys.LOAD_WORKFLOW: "workflows/MAGIC_PRINT",
    }


def without_key(a_dict, key):
    return {index: value for index, value in a_dict.config_data.items() if index != key}


def test_different_hook_workflow_gives_not_equal_hook_managers(
    workflow_example, workflow_example_dict
):
    assert HookManager(
        workflow_list=workflow_example.ert_workflow_list,
        config_dict=workflow_example_dict,
    ) != HookManager(
        workflow_list=workflow_example.ert_workflow_list,
        config_dict=without_key(workflow_example_dict, ConfigKeys.HOOK_WORKFLOW_KEY),
    )


def test_old_and_new_constructor_creates_equal_config(
    workflow_example, workflow_example_dict
):
    assert workflow_example.hook_manager == HookManager(
        workflow_list=workflow_example.ert_workflow_list,
        config_dict=workflow_example_dict,
    )


def test_all_config_entries_are_set(workflow_example, workflow_example_dict):
    hook_manager = HookManager(
        workflow_list=workflow_example.ert_workflow_list,
        config_dict=workflow_example_dict,
    )

    assert len(hook_manager) == 1

    magic_workflow = hook_manager[0]
    conf_dir = workflow_example_dict[ConfigKeys.CONFIG_DIRECTORY]
    assert magic_workflow.getWorkflow().src_file == os.path.join(
        conf_dir, workflow_example_dict[ConfigKeys.LOAD_WORKFLOW]
    )
    assert magic_workflow.getRunMode() == HookRuntime.PRE_SIMULATION
