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
from ecl.util.test import TestArea
from res.enkf import ErtTemplates
from res.enkf import ConfigKeys
from res.enkf import ResConfig
import unittest
from tests import ResTest
import os
import copy


class ErtTemplatesTest(ResTest):

    def setUp(self):
        self.work_area = TestArea("ert_templates_test_tmp")
        self.config_data = {
            ConfigKeys.CONFIG_DIRECTORY: self.work_area.get_cwd(),
            ConfigKeys.CONFIG_FILE_KEY: "config",
            ConfigKeys.RUN_TEMPLATE: [
                ("namea", "filea", [("keyaa", "valaa"), ("keyab", "valab"), ("keyac", "valac")]),
                ("nameb", "fileb", [("keyba", "valba"), ("keybb", "valbb"), ("keybc", "valbc")]),
                ("namec", "filec", [("keyca", "valca"), ("keycb", "valcb"), ("keycc", "valcc")])
            ]
        }
        self.filename = self.config_data[ConfigKeys.CONFIG_FILE_KEY]

        # write a config file in order to load ResConfig
        self.make_config_file(self.filename)
        self.make_empty_file("namea")
        self.make_empty_file("nameb")
        self.make_empty_file("namec")
        self.res_config = ResConfig(user_config_file=self.filename)

    def tearDown(self):
        del self.work_area

    def test_all_templates_exist_with_correct_properties(self):
        templates = ErtTemplates(self.res_config.subst_config.subst_list, config_dict=self.config_data)
        template_names = templates.getTemplateNames()
        configured_target_files = [t[0] for t in self.config_data[ConfigKeys.RUN_TEMPLATE]]
        assert set(template_names) == set(configured_target_files)
        for configured_template in self.config_data[ConfigKeys.RUN_TEMPLATE]:
            template = templates.get_template(configured_template[0])
            self.assertEqual(template.get_template_file(), os.path.join(self.work_area.get_cwd(), configured_template[0]))
            self.assertEqual(template.get_target_file(), configured_template[1])
            expected_arg_string = ", ".join(["{}={}".format(key, val) for key, val in configured_template[2]])
            self.assertEqual(expected_arg_string, template.get_args_as_string())

    def test_old_and_new_logic_produces_equal_objects(self):
        templates = ErtTemplates(self.res_config.subst_config.subst_list, config_dict=self.config_data)
        self.assertEqual(templates, self.res_config.ert_templates)

    def test_unequal_objects_are_unequal(self):
        templates = ErtTemplates(self.res_config.subst_config.subst_list,
                                 config_dict=self.config_data)
        templates2 = ErtTemplates(self.res_config.subst_config.subst_list,
                                  config_dict=self.change_template_arg(1, 1, "XXX", "YYY"))
        self.assertNotEqual(templates, templates2)

    def remove_key(self, key):
        return {i: self.config_data[i] for i in self.config_data if i != key}

    def change_template_arg(self, template_index, arg_index, new_key, new_val):
        conf_copy = copy.deepcopy(self.config_data)
        conf_copy[ConfigKeys.RUN_TEMPLATE][template_index][2][arg_index] = (new_key, new_val)
        return conf_copy

    def make_config_file(self, filename):
        with open(filename, "w+") as config:
            # necessary in the file, but irrelevant to this test
            config.write("JOBNAME  Job%d\n")
            config.write("NUM_REALIZATIONS  1\n")

            for template, target, args in self.config_data[ConfigKeys.RUN_TEMPLATE]:
                argstring = " ".join("{}:{}".format(key, val) for key, val in args)
                config.write("{} {} {} {}\n".format(ConfigKeys.RUN_TEMPLATE, template, target, argstring))

    def make_empty_file(self, filename):
        open(filename, 'a').close()


if __name__ == '__main__':
    unittest.main()
