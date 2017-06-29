#  Copyright (C) 2017  Statoil ASA, Norway.
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

from ecl.test import ExtendedTestCase, TestAreaContext

from res.enkf import ResConfig, SiteConfig, AnalysisConfig

config_defines = {
        "<USER>"         : "TEST_USER",
        "<SCRATCH>"      : "scratch/ert",
        "<CASE_DIR>"     : "the_extensive_case",
        "<ECLIPSE_NAME>" : "XYZ"
        }

config_data = {
        "RUNPATH"          : "<SCRATCH>/<USER>/<CASE_DIR>/realization-%d/iter-%d",
        "NUM_REALIZATIONS" : 10,
        "MAX_RUNTIME"      : 23400,
        "MIN_REALIZATIONS" : "50%",
        "MAX_SUBMIT"       : 13
        }

def expand_config_data():
    for define_key in config_defines:
        for data_key in config_data:
            if type(config_data[data_key]) is str:
                config_data[data_key] = config_data[data_key].replace(
                                                        define_key,
                                                        config_defines[define_key]
                                                        )

class ResConfigTest(ExtendedTestCase):

    def set_up_simple(self):
        self.case_directory = self.createTestPath("local/simple_config/")

    def set_up_snake_oil_structure(self):
        self.case_directory = self.createTestPath("local/snake_oil_structure")
        self.config_file = "snake_oil_structure/ert/model/user_config.ert"
        expand_config_data()

    def test_invalid_user_config(self):
        self.set_up_simple()

        with TestAreaContext("void land"):
            with self.assertRaises(IOError):
                ResConfig("this/is/not/a/file")

    def test_init(self):
        self.set_up_simple()

        with TestAreaContext("res_config_init_test") as work_area:
            cwd = os.getcwd()
            work_area.copy_directory(self.case_directory)

            config_file = "simple_config/minimum_config"
            res_config = ResConfig(user_config_file=config_file)

            self.assertIsNotNone(res_config)
            self.assertEqual(config_file, res_config.user_config_file)

            self.assertIsNotNone(res_config.site_config)
            self.assertTrue(isinstance(res_config.site_config, SiteConfig))

            self.assertIsNotNone(res_config.analysis_config)
            self.assertTrue(isinstance(res_config.analysis_config, AnalysisConfig))

            self.assertEqual( res_config.config_path , os.path.join( cwd , "simple_config"))

            config_file = os.path.join( cwd, "simple_config/minimum_config")
            res_config = ResConfig(user_config_file=config_file)
            self.assertEqual( res_config.config_path , os.path.join( cwd , "simple_config"))

            os.chdir("simple_config")
            config_file = "minimum_config"
            res_config = ResConfig(user_config_file=config_file)
            self.assertEqual( res_config.config_path , os.path.join( cwd , "simple_config"))

            subst_config = res_config.subst_config
            for t in subst_config:
                print t
            self.assertEqual( subst_config["<CONFIG_PATH>"], os.path.join( cwd , "simple_config"))

    def test_extensive_config(self):
        self.set_up_snake_oil_structure()

        with TestAreaContext("enkf_test_other_area") as work_area:
            work_area.copy_directory(self.case_directory)

            # Move to another directory
            run_dir = "i/ll/camp/here"
            os.makedirs(run_dir)
            os.chdir(run_dir)

            rel_config_file = "/".join(
                                   [".."] * len(run_dir.split("/")) +
                                   [self.config_file]
                                   )

            res_config = ResConfig(rel_config_file)

            # Test properties
            self.assertEqual(
                    config_data["RUNPATH"],
                    res_config.model_config.getRunpathAsString()
                    )

            self.assertEqual(
                    config_data["MAX_RUNTIME"],
                    res_config.analysis_config.get_max_runtime()
                    )

            self.assertEqual(
                    config_data["MAX_SUBMIT"],
                    res_config.site_config.queue_config.max_submit
                    )

            # TODO: Not tested
            # - NUM_REALIZATIONS
            # - MIN_REALIZATIONS
