import os

import pytest

from ert.config import ConfigValidationError, ErtConfig
from ert.config.parsing import ConfigKeys


def test_default_num_cpu():
    ert_config = ErtConfig.from_file_contents("NUM_REALIZATIONS 1")
    assert ert_config.preferred_num_cpu == 1


@pytest.mark.usefixtures("use_tmpdir")
def test_num_cpu_from_config_preferred():
    data_file = "dfile"
    config_num_cpu = 17
    data_file_num_cpu = 4
    with open(file=data_file, mode="w", encoding="utf-8") as data_file_hander:
        data_file_hander.write(
            f"""PARALLEL
 {data_file_num_cpu} DISTRIBUTED/
"""
        )
    config_dict = {
        ConfigKeys.NUM_REALIZATIONS: 1,
        ConfigKeys.NUM_CPU: config_num_cpu,
        ConfigKeys.DATA_FILE: os.path.join(os.getcwd(), data_file),
        ConfigKeys.ENSPATH: ".",
        ConfigKeys.RUNPATH_FILE: os.path.join(os.getcwd(), "runpath.file"),
    }
    ert_config = ErtConfig.from_dict(config_dict)
    assert ert_config.preferred_num_cpu == config_num_cpu


@pytest.mark.usefixtures("use_tmpdir")
def test_num_cpu_from_data_file_used_if_config_num_cpu_not_set():
    data_file = "dfile"
    data_file_num_cpu = 4
    with open(file=data_file, mode="w", encoding="utf-8") as data_file_hander:
        data_file_hander.write(
            f"""
PARALLEL
 {data_file_num_cpu} DISTRIBUTED/
"""
        )
    config_dict = {
        ConfigKeys.NUM_REALIZATIONS: 1,
        ConfigKeys.DATA_FILE: data_file,
        ConfigKeys.ENSPATH: ".",
        ConfigKeys.RUNPATH_FILE: os.path.join(os.getcwd(), "runpath.file"),
    }
    ert_config = ErtConfig.from_dict(config_dict)
    assert ert_config.preferred_num_cpu == data_file_num_cpu


@pytest.mark.parametrize(
    "num_cpu_value, error_msg",
    [
        (-1, "must have a positive integer value as argument"),
        (0, "must have a positive integer value as argument"),
        (1.5, "must have an integer value as argument"),
    ],
)
def test_wrong_num_cpu_raises_validation_error(num_cpu_value, error_msg):
    with pytest.raises(ConfigValidationError, match=error_msg):
        ErtConfig.from_file_contents(
            f"{ConfigKeys.NUM_REALIZATIONS} 1\n"
            f"{ConfigKeys.NUM_CPU} {num_cpu_value}\n"
        )
