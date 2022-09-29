import pytest

from ert._c_wrappers.enkf import ConfigKeys, EnKFMain, ResConfig


@pytest.mark.usefixtures("use_tmpdir")
def test_default_num_cpu():
    config_dict = {ConfigKeys.NUM_REALIZATIONS: 1}
    res_config = ResConfig(config_dict=config_dict)
    enkf_main = EnKFMain(res_config)
    assert enkf_main.get_num_cpu() == 1


@pytest.mark.usefixtures("use_tmpdir")
def test_num_cpu_from_queue_config_preferred():
    data_file = "dfile"
    queue_num_cpu = 17
    ecl_num_cpu = 4
    with open(data_file, "w") as data_file_hander:
        data_file_hander.write(
            f"""
PARALLEL
 {ecl_num_cpu} DISTRIBUTED/
"""
        )
    config_dict = {
        ConfigKeys.NUM_REALIZATIONS: 1,
        ConfigKeys.NUM_CPU: queue_num_cpu,
        ConfigKeys.DATA_FILE: data_file,
    }
    res_config = ResConfig(config_dict=config_dict)
    enkf_main: EnKFMain = EnKFMain(res_config)
    assert enkf_main.get_queue_config().num_cpu == queue_num_cpu
    assert enkf_main.eclConfig().num_cpu == ecl_num_cpu
    assert queue_num_cpu == enkf_main.get_num_cpu()


@pytest.mark.usefixtures("use_tmpdir")
def test_num_cpu_from_ecl_used_if_queue_num_cpu_not_set():
    data_file = "dfile"
    default_queue_num_cpu = 0
    ecl_num_cpu = 4
    with open(data_file, "w") as data_file_hander:
        data_file_hander.write(
            f"""
PARALLEL
 {ecl_num_cpu} DISTRIBUTED/
"""
        )
    config_dict = {
        ConfigKeys.NUM_REALIZATIONS: 1,
        ConfigKeys.DATA_FILE: data_file,
    }
    res_config = ResConfig(config_dict=config_dict)
    enkf_main: EnKFMain = EnKFMain(res_config)
    assert enkf_main.get_queue_config().num_cpu == default_queue_num_cpu
    assert enkf_main.eclConfig().num_cpu == ecl_num_cpu
    assert ecl_num_cpu == enkf_main.get_num_cpu()
