import os

import pytest
from hypothesis import given

from ert._c_wrappers.enkf import ConfigKeys, QueueConfig

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
