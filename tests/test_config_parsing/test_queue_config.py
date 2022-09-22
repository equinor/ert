import os

import pytest
from hypothesis import given, settings, reproduce_failure

from ert._c_wrappers.enkf import ConfigKeys, QueueConfig, ResConfig

from .config_dict_generator import config_dicts, to_config_file


@pytest.mark.skip(reason="https://github.com/equinor/ert/issues/2571")
@pytest.mark.usefixtures("use_tmpdir")
@given(config_dicts())
def test_queue_config_dict_same_as_from_file(config_dict):
    cwd = os.getcwd()
    filename = config_dict[ConfigKeys.CONFIG_FILE_KEY]
    to_config_file(filename, config_dict)
    config_dict[ConfigKeys.CONFIG_DIRECTORY] = cwd
    assert QueueConfig(filename) == QueueConfig(config_dict=config_dict)


@pytest.mark.usefixtures("use_tmpdir")
@given(config_dicts())
@settings(print_blob=True)
@reproduce_failure("6.54.5", b"AXicY2BkgEAGOIMBlQ0FjNjEsAmhSDEiq8KinJAsXgAAFBgALA==")
def test_queue_config_dict_works_without_num_cpu(config_dict):
    filename = config_dict[ConfigKeys.CONFIG_FILE_KEY]
    cwd = os.getcwd()
    config_dict[ConfigKeys.CONFIG_DIRECTORY] = cwd
    config_dict.pop(ConfigKeys.NUM_CPU)
    to_config_file(filename, config_dict)
    queue_config_from_file = ResConfig(filename).queue_config
    assert queue_config_from_file.num_cpu == -1
