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
from res.enkf import QueueConfig, ConfigKeys
from res.config import ConfigContent

class QueueConfigTest(ResTest):
    def setUp(self):
        self.case_directory = self.createTestPath("local/simple_config/")

    def test_get_queue_config(self):
        with TestAreaContext("queue_config_init_test") as work_area:
            work_area.copy_directory(self.case_directory)

            config_file = "simple_config/minimum_config"
            queue_config = QueueConfig(config_file)
            job_queue = queue_config.create_job_queue()
            queue_config_copy = queue_config.create_local_copy()

            self.assertEqual(
                    queue_config.has_job_script(),
                    queue_config_copy.has_job_script()
                    )


            config_content = ConfigContent(config_file)

            with self.assertRaises(ValueError):
                queue_config = QueueConfig(user_config_file=config_file, config_content=config_content)

    def test_queue_config_constructor(self):
        with TestAreaContext("queue_config_constructor_test") as work_area:
            work_area.copy_directory(self.case_directory)
            config_file = "simple_config/minimum_config"
            config_dict = {
                ConfigKeys.JOB_SCRIPT: os.getcwd() + "/simple_config/script.sh",
                ConfigKeys.QUEUE_SYSTEM: 2,
                ConfigKeys.USER_MODE: True,
                ConfigKeys.MAX_SUBMIT: 2,
                ConfigKeys.NUM_CPU: 0,
                ConfigKeys.QUEUE_OPTION: [
                    {
                        ConfigKeys.NAME: "MAX_RUNNING",
                        ConfigKeys.VALUE: "50"
                    }
                ]
            }

            queue_config_file = QueueConfig(user_config_file=config_file)
            queue_config_dict = QueueConfig(config_dict=config_dict)
            self.assertEqual(queue_config_dict, queue_config_file)

    def test_get_slurm_queue_config(self):
        with TestAreaContext("queue_config_slurm_test") as work_area:
            work_area.copy_directory(self.case_directory)

            config_file = "simple_config/slurm_config"
            queue_config = QueueConfig(config_file)
            self.assertEqual(queue_config.queue_system, "SLURM")

            driver = queue_config.driver
            self.assertEqual(driver.get_option("SBATCH"), "/path/to/sbatch")
            self.assertEqual(driver.get_option("SCONTROL"), "scontrol")
            self.assertEqual(driver.name, "SLURM")
