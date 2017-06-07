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

from ecl.test import ExtendedTestCase, TestAreaContext

from res.enkf import ResConfig, SiteConfig, AnalysisConfig

class ResConfigTest(ExtendedTestCase):

    def setUp(self):
        self.case_directory = self.createTestPath("local/simple_config/")

    def test_invalid_user_config(self):
        with TestAreaContext("void land"):
            with self.assertRaises(IOError):
                ResConfig("this/is/not/a/file")

    def test_init(self):
        with TestAreaContext("res_config_init_test") as work_area:
            work_area.copy_directory(self.case_directory)

            config_file = "simple_config/minimum_config"
            res_config = ResConfig(user_config_file=config_file)

            self.assertIsNotNone(res_config)
            self.assertEqual(config_file, res_config.user_config_file)

            self.assertIsNotNone(res_config.site_config)
            self.assertTrue(isinstance(res_config.site_config, SiteConfig))

            self.assertIsNotNone(res_config.analysis_config)
            self.assertTrue(isinstance(res_config.analysis_config, AnalysisConfig))
