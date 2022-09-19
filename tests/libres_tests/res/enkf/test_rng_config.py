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

import pytest

from ert._c_wrappers.enkf import ConfigKeys, ResConfig, RNGConfig


@pytest.mark.usefixtures("copy_minimum_case")
def test_compare_config_content_and_dict_constructor():
    assert ResConfig(
        config={
            "INTERNALS": {
                "CONFIG_DIRECTORY": ".",
            },
            "SIMULATION": {
                "RANDOM_SEED": "abcdefghijklmnop",
                "QUEUE_SYSTEM": {
                    "JOBNAME": "Job%d",
                },
                "RUNPATH": "/tmp/simulations/run%d",
                "NUM_REALIZATIONS": 1,
                "JOB_SCRIPT": "script.sh",
                "ENSPATH": "Ensemble",
            },
        }
    ).rng_config == RNGConfig(config_dict={ConfigKeys.RANDOM_SEED: "abcdefghijklmnop"})


def test_random_seed():
    assert (
        RNGConfig(config_dict={ConfigKeys.RANDOM_SEED: "abcdefghijklmnop"}).random_seed
        == "abcdefghijklmnop"
    )


@pytest.mark.usefixtures("copy_minimum_case")
def test_default_random_seed():
    assert ResConfig(
        config={
            "INTERNALS": {
                "CONFIG_DIRECTORY": ".",
            },
            "SIMULATION": {
                "QUEUE_SYSTEM": {
                    "JOBNAME": "Job%d",
                },
                "RUNPATH": "/tmp/simulations/run%d",
                "NUM_REALIZATIONS": 1,
                "JOB_SCRIPT": "script.sh",
                "ENSPATH": "Ensemble",
            },
        }
    ).rng_config == RNGConfig(config_dict={ConfigKeys.RANDOM_SEED: None})
