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
import os.path

from ert._c_wrappers.enkf import QueueConfig
from ert._c_wrappers.job_queue import QueueDriverEnum


def test_get_queue_config(minimum_case):
    queue_config = minimum_case.resConfig().queue_config
    queue_config.create_job_queue()
    queue_config_copy = queue_config.create_local_copy()
    assert queue_config_copy.queue_system == QueueDriverEnum.LOCAL_DRIVER


def test_queue_config_constructor(minimum_case):
    queue_config_relative = QueueConfig(
        job_script="script.sh",
        queue_system=QueueDriverEnum(2),
        max_submit=2,
        num_cpu=0,
        queue_options={
            QueueDriverEnum.LOCAL_DRIVER: [
                ("MAX_RUNNING", "1"),
                ("MAX_RUNNING", "50"),
            ]
        },
    )

    queue_config_absolute = QueueConfig(
        job_script=os.path.abspath("script.sh"),
        queue_system=QueueDriverEnum(2),
        max_submit=2,
        num_cpu=0,
        queue_options={
            QueueDriverEnum.LOCAL_DRIVER: [
                ("MAX_RUNNING", "1"),
                ("MAX_RUNNING", "50"),
            ]
        },
    )
    minimum_queue_config = minimum_case.resConfig().queue_config

    # Depends on where you run the tests
    assert minimum_queue_config in (queue_config_absolute, queue_config_relative)


def test_set_and_unset_option():
    queue_config = QueueConfig(
        job_script="script.sh",
        queue_system=QueueDriverEnum(2),
        max_submit=2,
        num_cpu=0,
        queue_options={
            QueueDriverEnum.LOCAL_DRIVER: [
                ("MAX_RUNNING", "50"),
                "MAX_RUNNING",
            ]
        },
    )
    assert queue_config.create_driver().get_option("MAX_RUNNING") == "0"


def test_get_slurm_queue_config(setup_case):
    res_config = setup_case("simple_config", "slurm_config")
    queue_config = res_config.queue_config

    assert queue_config.queue_system == QueueDriverEnum.SLURM_DRIVER
    driver = queue_config.create_driver()
    assert driver.get_option("SBATCH") == "/path/to/sbatch"
    assert driver.get_option("SCONTROL") == "scontrol"
    assert driver.name == "SLURM"
