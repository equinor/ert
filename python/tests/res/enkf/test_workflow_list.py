#  Copyright (C) 2017  Equinor ASA, Norway.
#
#  The file 'test_summary_obs.py' is part of ERT - Ensemble based Reservoir Tool.
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
from ecl.util.test import TestAreaContext
from tests import ResTest
from res.enkf import ErtWorkflowList, ResConfig, SiteConfig


class ErtWorkflowListTest(ResTest):
    def setUp(self):
        self.case_directory = self.createTestPath("local/simple_config/")
        

    def test_workflow_list_constructor(self):
        with TestAreaContext("ert_workflow_list_test") as work_area:

            ERT_SITE_CONFIG = SiteConfig.getLocation()
            ERT_SHARE_PATH = os.path.dirname(ERT_SITE_CONFIG)

            self.config_dict = {
            "LOAD_WORKFLOW_JOB":
                [
                    {
                        "NAME" : "print_uber",
                        "PATH" : os.getcwd() + "/simple_config/workflows/UBER_PRINT"
                    }
                ],

            "LOAD_WORKFLOW":
                [
                    {
                        "NAME" : "magic_print",
                        "PATH" : os.getcwd() + "/simple_config/workflows/MAGIC_PRINT"
                    }
                ],
            'WORKFLOW_JOB_DIRECTORY': [
                    ERT_SHARE_PATH + '/workflows/jobs/internal/config',
                    ERT_SHARE_PATH + '/workflows/jobs/internal-gui/config'
                ],
            }

            work_area.copy_directory(self.case_directory)

            config_file = "simple_config/minimum_config"
            with open(config_file,'a+') as ert_file:
                ert_file.write("LOAD_WORKFLOW_JOB workflows/UBER_PRINT print_uber\n")
                ert_file.write("LOAD_WORKFLOW workflows/MAGIC_PRINT magic_print\n")

            os.mkdir("simple_config/workflows")

            with open("simple_config/workflows/MAGIC_PRINT","w") as f:
                f.write("print_uber\n")
            with open("simple_config/workflows/UBER_PRINT", "w") as f:
                f.write("EXECUTABLE ls\n")
            
            res_config = ResConfig(config_file)
            list_from_content = res_config.ert_workflow_list
            list_from_dict = ErtWorkflowList(subst_list=res_config.subst_config.subst_list, config_dict=self.config_dict)
            self.assertEqual(list_from_content, list_from_dict)


    def test_illegal_configs(self):
        with self.assertRaises(ValueError):
            ErtWorkflowList(subst_list=[], config_dict=[], config_content=[])

        with self.assertRaises(ValueError):
            ErtWorkflowList()
