import os

import pytest
from hypothesis import given

from ert._c_wrappers.enkf import ConfigKeys, ResConfig

from .config_dict_generator import config_dicts, to_config_file


@pytest.mark.usefixtures("use_tmpdir", "set_site_config")
@given(config_dicts())
def test_queue_config_dict_same_as_from_file(config_dict):
    cwd = os.getcwd()
    filename = config_dict[ConfigKeys.CONFIG_FILE_KEY]
    to_config_file(filename, config_dict)
    config_dict[ConfigKeys.CONFIG_DIRECTORY] = cwd
    assert (
        ResConfig(filename).queue_config
        == ResConfig(config_dict=config_dict).queue_config
    )


@pytest.mark.usefixtures("use_tmpdir")
@given(config_dicts())
def test_queue_config_dict_works_without_num_cpu(config_dict):
    filename = config_dict[ConfigKeys.CONFIG_FILE_KEY]
    cwd = os.getcwd()
    config_dict[ConfigKeys.CONFIG_DIRECTORY] = cwd
    config_dict.pop(ConfigKeys.NUM_CPU)
    to_config_file(filename, config_dict)
    config_dict[ConfigKeys.CONFIG_DIRECTORY] = cwd

    queue_config_from_dict = ResConfig(config_dict=config_dict).queue_config

    assert queue_config_from_dict.num_cpu == 0
