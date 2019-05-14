#  Copyright (C) 2017  Equinor ASA, Norway.
#
#  The file 'test_site_config.py' is part of ERT - Ensemble based Reservoir Tool.
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

from res.enkf import SiteConfig
import os

from ecl.util.test import TestAreaContext
from tests import ResTest

class SiteConfigTest(ResTest):

    def setUp(self):
        self.case_directory = self.createTestPath("local/simple_config/")
        self.snake_case_directory = self.createTestPath("local/snake_oil/")

    def test_invalid_user_config(self):
        with TestAreaContext("void land"):
            with self.assertRaises(IOError):
                SiteConfig("this/is/not/a/file")

    def test_init(self):
        with TestAreaContext("site_config_init_test") as work_area:
            work_area.copy_directory(self.case_directory)
            config_file = "simple_config/minimum_config"
            site_config = SiteConfig(user_config_file=config_file)
            self.assertIsNotNone(site_config)

    def test_constructors(self):
        with TestAreaContext("site_config_constructor_test") as work_area:
            work_area.copy_directory(self.snake_case_directory)
            config_file = "snake_oil/snake_oil.ert"

            ERT_SITE_CONFIG = SiteConfig.getLocation()
            ERT_SHARE_PATH = os.path.dirname(ERT_SITE_CONFIG)
            snake_config_dict = {
                "INSTALL_JOB":
                    [
                        {
                            "NAME": "SNAKE_OIL_SIMULATOR",
                            "PATH": os.getcwd() + "/snake_oil/jobs/SNAKE_OIL_SIMULATOR"
                        },
                        {
                            "NAME": "SNAKE_OIL_NPV",
                            "PATH": os.getcwd() + "/snake_oil/jobs/SNAKE_OIL_NPV"
                        },
                        {
                            "NAME": "SNAKE_OIL_DIFF",
                            "PATH": os.getcwd() + "/snake_oil/jobs/SNAKE_OIL_DIFF"
                        }
                    ],
                "INSTALL_JOB_DIRECTORY":
                    [
                        ERT_SHARE_PATH + '/forward-models/res',
                        ERT_SHARE_PATH + '/forward-models/shell',
                        ERT_SHARE_PATH + '/forward-models/templating'
                    ],

                "SETENV":
                    [
                        {
                            "NAME": "SILLY_VAR",
                            "VALUE": "silly-value"
                        },
                        {
                            "NAME": "OPTIONAL_VAR",
                            "VALUE": "optional-value"
                        }
                    ],
                "LICENSE_PATH": "some/random/path",

                "UMASK": 18
            }

            site_config_user_file = SiteConfig(user_config_file=config_file)
            site_config_dict = SiteConfig(config_dict=snake_config_dict)
            self.assertEqual(site_config_dict, site_config_user_file)

            with self.assertRaises(ValueError):
                site_config = SiteConfig(user_config_file=config_file, config_dict=snake_config_dict)