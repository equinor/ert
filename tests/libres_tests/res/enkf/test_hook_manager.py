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
import unittest

from ecl.util.test import TestArea
from ert._c_wrappers.enkf import ConfigKeys, HookManager, HookRuntime, ResConfig

from ...libres_utils import ResTest


class HookManagerTest(ResTest):
    def setUp(self):
        self.work_area = TestArea("hook_manager_test_tmp")
        # in order to test HOOK_WORKFLOWS there need to be some workflows loaded
        self.work_area.copy_directory(
            self.createTestPath("local/config/workflows/workflowjobs")
        )
        self.work_area.copy_directory(
            self.createTestPath("local/config/workflows/workflows")
        )
        self.config_data = {
            ConfigKeys.RUNPATH_FILE: "runpath",
            ConfigKeys.CONFIG_DIRECTORY: self.work_area.get_cwd(),
            ConfigKeys.CONFIG_FILE_KEY: "config",
            ConfigKeys.HOOK_WORKFLOW_KEY: [("MAGIC_PRINT", "PRE_SIMULATION")],
            # these two entries makes the workflow_list load this workflow, but
            # are not needed by hook_manager directly
            ConfigKeys.LOAD_WORKFLOW_JOB: "workflowjobs/MAGIC_PRINT",
            ConfigKeys.LOAD_WORKFLOW: "workflows/MAGIC_PRINT",
        }
        self.filename = self.config_data[ConfigKeys.CONFIG_FILE_KEY]
        # these files must exist
        self.make_empty_file(self.config_data[ConfigKeys.RUNPATH_FILE])

        # write a config file in order to load ResConfig
        self.make_config_file(self.filename)
        self.res_config = ResConfig(user_config_file=self.filename)

    def tearDown(self):
        del self.work_area

    def test_different_hook_workflow_gives_not_equal_hook_managers(self):
        res_config2 = ResConfig(user_config_file=self.filename)
        hook_manager1 = HookManager(
            workflow_list=self.res_config.ert_workflow_list,
            config_dict=self.config_data,
        )
        hook_manager2 = HookManager(
            workflow_list=res_config2.ert_workflow_list,
            config_dict=self.remove_key(ConfigKeys.HOOK_WORKFLOW_KEY),
        )

        self.assertNotEqual(hook_manager1, hook_manager2)

    def test_old_and_new_constructor_creates_equal_config(self):
        res_config2 = ResConfig(user_config_file=self.filename)
        old = res_config2.hook_manager
        new = HookManager(
            workflow_list=self.res_config.ert_workflow_list,
            config_dict=self.config_data,
        )

        self.assertEqual(old, new)

    def test_all_config_entries_are_set(self):
        hook_manager = HookManager(
            workflow_list=self.res_config.ert_workflow_list,
            config_dict=self.config_data,
        )

        self.assertEqual(len(hook_manager), 1)

        magic_workflow = hook_manager[0]
        conf_dir = self.config_data[ConfigKeys.CONFIG_DIRECTORY]
        self.assertEqual(
            magic_workflow.getWorkflow().src_file,
            os.path.join(conf_dir, self.config_data[ConfigKeys.LOAD_WORKFLOW]),
        )
        self.assertEqual(magic_workflow.getRunMode(), HookRuntime.PRE_SIMULATION)

    def remove_key(self, key):
        return {
            index: value for index, value in self.config_data.items() if index != key
        }

    def set_key(self, key, val):
        copy = self.config_data.copy()
        copy[key] = val
        return copy

    def make_config_file(self, filename):
        with open(filename, "w+") as config:
            # necessary in the file, but irrelevant to this test
            config.write("JOBNAME  Job%d\n")
            config.write("NUM_REALIZATIONS  1\n")
            for key, val in self.config_data.items():
                if key in (ConfigKeys.CONFIG_FILE_KEY, ConfigKeys.CONFIG_DIRECTORY):
                    continue
                if isinstance(val, str):
                    config.write(f"{key} {val}\n")
                else:
                    # assume this is the list of tuple for hook workflows
                    for val1, val2 in val:
                        config.write(f"{key} {val1} {val2}\n")

    def make_empty_file(self, filename):
        open(filename, "a").close()


if __name__ == "__main__":
    unittest.main()
