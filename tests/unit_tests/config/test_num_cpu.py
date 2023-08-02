import os

import pytest

from ert.config import ErtConfig
from ert.config.parsing import ConfigKeys
from ert.enkf_main import EnKFMain


@pytest.mark.usefixtures("use_tmpdir")
def test_default_num_cpu():
    with open("file.ert", mode="w", encoding="utf-8") as f:
        f.write(f"{ConfigKeys.NUM_REALIZATIONS} 1")
    ert_config = ErtConfig.from_file("file.ert")
    enkf_main = EnKFMain(ert_config)
    assert enkf_main.get_num_cpu() == 1


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
    enkf_main = EnKFMain(ert_config)
    assert ert_config.preferred_num_cpu() == config_num_cpu
    assert enkf_main.get_num_cpu() == config_num_cpu


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
    enkf_main = EnKFMain(ert_config)
    assert enkf_main.resConfig().preferred_num_cpu() == data_file_num_cpu
    assert enkf_main.get_num_cpu() == data_file_num_cpu
