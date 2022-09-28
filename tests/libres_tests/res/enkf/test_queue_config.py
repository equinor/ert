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

from ert._c_wrappers.enkf import QueueConfig
from ert._c_wrappers.job_queue import QueueDriverEnum


def test_get_queue_config(minimum_case):
    queue_config = minimum_case.resConfig().queue_config
    queue_config.create_job_queue()
    queue_config_copy = queue_config.create_local_copy()
    assert queue_config_copy.queue_system == "LOCAL"


def test_queue_config_constructor(minimum_case):
    assert (
        QueueConfig(
            queue_system=QueueDriverEnum(2),
            max_submit=2,
            num_cpu=0,
        )
        == minimum_case.resConfig().queue_config
    )


def test_get_slurm_queue_config(setup_case):
    res_config = setup_case("local/simple_config", "slurm_config")
    queue_config = res_config.queue_config

    assert queue_config.queue_system == "SLURM"
    driver = queue_config.driver
    assert driver.get_option("SBATCH") == "/path/to/sbatch"
    assert driver.get_option("SCONTROL") == "scontrol"
    assert driver.name == "SLURM"
