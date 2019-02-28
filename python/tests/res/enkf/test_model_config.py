#  Copyright (C) 2017  Equinor ASA, Norway.
#
#  The file 'test_model_config.py' is part of ERT - Ensemble based Reservoir Tool.
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
from res.test import ErtTestContext
from res.enkf import ResConfig, ConfigKeys

class ModelConfigTest(ResTest):

    def setUp(self):
        self.config_both = {
                                "INTERNALS" :
                                {
                                  "CONFIG_DIRECTORY" : "simple_config",
                                },

                                "SIMULATION" :
                                {
                                  "QUEUE_SYSTEM" :
                                  {
                                    "JOBNAME" : "JOBNAME%d",
                                  },

                                  "RUNPATH"            : "/tmp/simulations/run%d",
                                  "NUM_REALIZATIONS"   : 1,
                                  "JOB_SCRIPT"         : "script.sh",
                                    "ENSPATH"          : "Ensemble",
                                  "ECLBASE"            : "ECLBASE%d"
                                }
                              }


        self.config_eclbase = {
                                "INTERNALS" :
                                {
                                  "CONFIG_DIRECTORY" : "simple_config",
                                },

                                "SIMULATION" :
                                {
                                  "RUNPATH"            : "/tmp/simulations/run%d",
                                  "NUM_REALIZATIONS"   : 1,
                                  "JOB_SCRIPT"         : "script.sh",
                                  "ENSPATH"            : "Ensemble",
                                  "ECLBASE"            : "ECLBASE%d"
                                }
                              }


        self.config_jobname = {
                                "INTERNALS" :
                                {
                                  "CONFIG_DIRECTORY" : "simple_config",
                                },

                                "SIMULATION" :
                                {
                                  "QUEUE_SYSTEM" :
                                  {
                                    "JOBNAME" : "JOBNAME%d",
                                  },

                                  "RUNPATH"            : "/tmp/simulations/run%d",
                                  "NUM_REALIZATIONS"   : 1,
                                  "JOB_SCRIPT"         : "script.sh",
                                  "ENSPATH"            : "Ensemble"
                                }
                              }



    def test_eclbase_and_jobname(self):
        case_directory = self.createTestPath("local/simple_config")
        with TestAreaContext("test_eclbase_and_jobname") as work_area:
            work_area.copy_directory(case_directory)

            res_config = ResConfig(config=self.config_both)
            model_config = res_config.model_config
            ecl_config = res_config.ecl_config

            self.assertTrue( ecl_config.active( ) )
            self.assertEqual( "JOBNAME%d" , model_config.getJobnameFormat())



    def test_eclbase(self):
        case_directory = self.createTestPath("local/simple_config")
        with TestAreaContext("test_eclbase") as work_area:
            work_area.copy_directory(case_directory)

            res_config = ResConfig(config=self.config_eclbase)
            model_config = res_config.model_config
            ecl_config = res_config.ecl_config

            self.assertTrue( ecl_config.active( ) )
            self.assertEqual( "ECLBASE%d" , model_config.getJobnameFormat( ))


    def test_jobname(self):
        case_directory = self.createTestPath("local/simple_config")
        with TestAreaContext("test_jobname") as work_area:
            work_area.copy_directory(case_directory)

            res_config = ResConfig(config=self.config_jobname)
            model_config = res_config.model_config
            ecl_config = res_config.ecl_config

            self.assertFalse( ecl_config.active( ) )
            self.assertEqual( "JOBNAME%d" , model_config.getJobnameFormat( ))

