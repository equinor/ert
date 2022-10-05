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

import pytest

from ert._c_wrappers.config import ConfigContent
from ert._c_wrappers.enkf import ConfigKeys, QueueConfig
from ert._c_wrappers.job_queue import QueueDriverEnum


def test_get_queue_config(minimum_case):
    queue_config = minimum_case.resConfig().queue_config
    queue_config.create_job_queue()
    queue_config_copy = queue_config.create_local_copy()

    assert queue_config.has_job_script() == queue_config_copy.has_job_script()


def test_that_qeueu_raises_given_two_init_params():
    with pytest.raises(ValueError, match="multiple config"):
        _ = QueueConfig(config_dict={}, config_content=ConfigContent("a_file"))


def test_queue_raises_given_no_init_params():
    with pytest.raises(ValueError, match="no config"):
        _ = QueueConfig()


def test_queue_config_constructor(minimum_case):
    assert (
        QueueConfig(
            config_dict={
                ConfigKeys.JOB_SCRIPT: os.getcwd() + "/script.sh",
                ConfigKeys.QUEUE_SYSTEM: QueueDriverEnum(2),
                ConfigKeys.USER_MODE: True,
                ConfigKeys.MAX_SUBMIT: 2,
                ConfigKeys.NUM_CPU: 0,
                ConfigKeys.QUEUE_OPTION: [
                    {
                        ConfigKeys.DRIVER_NAME: QueueDriverEnum(2),
                        ConfigKeys.OPTION: "MAX_RUNNING",
                        ConfigKeys.VALUE: "50",
                    }
                ],
            }
        )
        == minimum_case.resConfig().queue_config
    )


def test_get_slurm_queue_config(setup_case):
    res_config = setup_case("simple_config", "slurm_config")
    queue_config = res_config.queue_config

    assert queue_config.queue_system == "SLURM"
    driver = queue_config.driver
    assert driver.get_option("SBATCH") == "/path/to/sbatch"
    assert driver.get_option("SCONTROL") == "scontrol"
    assert driver.name == "SLURM"
