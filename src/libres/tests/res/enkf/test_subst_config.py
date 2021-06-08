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
from ecl.util.test import TestAreaContext
from res.enkf import SubstConfig
from res.enkf import ConfigKeys
from res.enkf import ResConfig
import unittest
from tests import ResTest


class SubstConfigTest(ResTest):
    def setUp(self):
        self.path = self.createTestPath("local/snake_oil_structure/")
        self.config_data = {
            ConfigKeys.RUNPATH_FILE: "runpath",
            ConfigKeys.CONFIG_DIRECTORY: self.path,
            ConfigKeys.CONFIG_FILE_KEY: "config",
            ConfigKeys.DEFINE_KEY: {"keyA": "valA", "keyB": "valB"},
            ConfigKeys.DATA_KW_KEY: {"keyC": "valC", "keyD": "valD"},
            ConfigKeys.DATA_FILE: "eclipse/model/SNAKE_OIL.DATA",
        }

    def test_two_instances_of_same_config_are_equal(self):
        subst_config1 = SubstConfig(config_dict=self.config_data)
        subst_config2 = SubstConfig(config_dict=self.config_data)
        self.assertEqual(subst_config1, subst_config2)

    def test_two_instances_of_different_config_are_not_equal(self):
        subst_config1 = SubstConfig(config_dict=self.config_data)
        subst_config2 = SubstConfig(
            config_dict=self.set_key(ConfigKeys.RUNPATH_FILE, "aaaaa")
        )
        self.assertNotEqual(subst_config1, subst_config2)

    def test_old_and_new_constructor_creates_equal_config(self):
        with TestAreaContext("subst_config_test_tmp") as work_area:
            work_area.copy_directory(os.path.join(self.path, "eclipse"))
            cwd = os.getcwd()
            filename = self.config_data[ConfigKeys.CONFIG_FILE_KEY]
            self.make_config_file(filename)
            res_config = ResConfig(user_config_file=filename)
            subst_config1 = res_config.subst_config
            subst_config2 = SubstConfig(
                config_dict=self.set_key(ConfigKeys.CONFIG_DIRECTORY, cwd)
            )

            self.assertEqual(
                subst_config1,
                subst_config2,
                str(subst_config1) + "\n\nis not equal to:\n\n" + str(subst_config2),
            )

    def test_complete_config_reads_correct_values(self):
        subst_config = SubstConfig(config_dict=self.config_data)
        self.assertKeyValue(subst_config, "<CWD>", self.path)
        self.assertKeyValue(subst_config, "<CONFIG_PATH>", self.path)
        self.assertKeyValue(subst_config, "keyA", "valA")
        self.assertKeyValue(subst_config, "keyB", "valB")
        self.assertKeyValue(subst_config, "keyC", "valC")
        self.assertKeyValue(subst_config, "keyD", "valD")
        self.assertKeyValue(subst_config, "<RUNPATH_FILE>", self.path + "/runpath")
        self.assertKeyValue(subst_config, "<NUM_CPU>", "1")

    def test_missing_runpath_gives_default_value(self):
        subst_config = SubstConfig(config_dict=self.remove_key(ConfigKeys.RUNPATH_FILE))
        self.assertKeyValue(
            subst_config, "<RUNPATH_FILE>", self.path + "/.ert_runpath_list"
        )

    def test_empty_config_raises_error(self):
        with self.assertRaises(ValueError):
            SubstConfig(config_dict={})

    def test_missing_config_directory_raises_error(self):
        with self.assertRaises(ValueError):
            SubstConfig(config_dict=self.remove_key(ConfigKeys.CONFIG_DIRECTORY))

    def test_data_file_not_found_raises_error(self):
        with self.assertRaises(IOError):
            SubstConfig(config_dict=self.set_key(ConfigKeys.DATA_FILE, "not_a_file"))

    def remove_key(self, key):
        return {i: self.config_data[i] for i in self.config_data if i != key}

    def set_key(self, key, val):
        copy = self.config_data.copy()
        copy[key] = val
        return copy

    def assertKeyValue(self, subst_config, key, val):
        actual_val = subst_config.__getitem__(key)
        assert (
            actual_val == val
        ), "subst_config does not contain key/value pair ({}, {}). Actual value was: {}".format(
            key, val, actual_val
        )

    def make_config_file(self, filename):
        with open(filename, "w+") as config:
            # necessary in the file, but irrelevant to this test
            config.write("JOBNAME  Job%d\n")
            config.write("NUM_REALIZATIONS  1\n")

            # write the rest of the relevant config items to the file
            config.write(
                "{} {}\n".format(
                    ConfigKeys.RUNPATH_FILE, self.config_data[ConfigKeys.RUNPATH_FILE]
                )
            )
            defines = self.config_data[ConfigKeys.DEFINE_KEY]
            for key in defines:
                val = defines[key]
                config.write("{} {} {}\n".format(ConfigKeys.DEFINE_KEY, key, val))
            data_kws = self.config_data[ConfigKeys.DATA_KW_KEY]
            for key in data_kws:
                val = data_kws[key]
                config.write("{} {} {}\n".format(ConfigKeys.DATA_KW_KEY, key, val))
            config.write(
                "{} {}\n".format(
                    ConfigKeys.DATA_FILE, self.config_data[ConfigKeys.DATA_FILE]
                )
            )


if __name__ == "__main__":
    unittest.main()
