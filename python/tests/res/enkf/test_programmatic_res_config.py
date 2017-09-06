#  Copyright (C) 2017  Statoil ASA, Norway.
#
#  The file 'test_programmatic_res_config.py' is part of ERT - Ensemble based Reservoir Tool.
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

from res.enkf import ResConfig

class ProgrammaticResConfigTest(ExtendedTestCase):

    minimum_config = {
                        "WORKING_DIRECTORY"  : "simple_config",
                        "JOBNAME"            : "Job%d",
                        "RUNPATH"            : "/tmp/simulations/run%d",
                        "NUM_REALIZATIONS"   : 1,
                        "JOB_SCRIPT"         : "script.sh",
                        "ENSPATH"            : "Ensemble"
                      }

    def test_minimum_config(self):
        case_directory = self.createTestPath("local/simple_config")
        config_file = "simple_config/minimum_config"

        with TestAreaContext("res_config_prog_test") as work_area:
            work_area.copy_directory(case_directory)

            loaded_res_config = ResConfig(user_config_file=config_file)
            prog_res_config = ResConfig(config=self.minimum_config)

            self.assertEqual(loaded_res_config.model_config.num_realizations,
                             prog_res_config.model_config.num_realizations)

            self.assertEqual(loaded_res_config.model_config.getJobnameFormat(),
                             prog_res_config.model_config.getJobnameFormat())

            self.assertEqual(loaded_res_config.model_config.getRunpathAsString(),
                             prog_res_config.model_config.getRunpathAsString())

            self.assertEqual(loaded_res_config.site_config.queue_config.job_script,
                             prog_res_config.site_config.queue_config.job_script)
