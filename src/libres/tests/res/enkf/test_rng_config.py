#  Copyright (C) 2017  Equinor ASA, Norway.
#
#  The file 'test_rng_config.py' is part of ERT - Ensemble based Reservoir Tool.
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

from ecl.util.test import TestAreaContext
from tests import ResTest

from res.enkf import ResConfig, RNGConfig, ConfigKeys


class RNGConfigTest(ResTest):
    def create_base_config(self):
        return {
            "INTERNALS": {
                "CONFIG_DIRECTORY": "simple_config",
            },
            "SIMULATION": {
                "QUEUE_SYSTEM": {
                    "JOBNAME": "Job%d",
                },
                "RUNPATH": "/tmp/simulations/run%d",
                "NUM_REALIZATIONS": 1,
                "JOB_SCRIPT": "script.sh",
                "ENSPATH": "Ensemble",
                "LOGGING": {"LOG_LEVEL": "DEBUG"},
            },
        }

    def test_dict_constructor(self):
        config = self.create_base_config()

        case_directory = self.createTestPath("local/simple_config")
        with TestAreaContext("rng_config") as work_area:
            work_area.copy_directory(case_directory)

            random_seed = "abcdefghijklmnop"
            config["SIMULATION"]["SEED"] = {ConfigKeys.RANDOM_SEED: random_seed}

            rng_config = RNGConfig(config_dict=config["SIMULATION"]["SEED"])
            res_config = ResConfig(config=config)
            self.assertEqual(rng_config, res_config.rng_config)

    def test_random_seed(self):
        config = self.create_base_config()

        random_seed = "abcdefghijklmnop"
        config["SIMULATION"]["SEED"] = {ConfigKeys.RANDOM_SEED: random_seed}

        case_directory = self.createTestPath("local/simple_config")
        with TestAreaContext("rng_config") as work_area:
            work_area.copy_directory(case_directory)
            res_config = ResConfig(config=config)

            self.assertEqual(random_seed, res_config.rng_config.random_seed)
