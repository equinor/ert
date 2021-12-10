import os
import os.path

import pytest
from hypothesis import given

from ert._c_wrappers.enkf import ConfigKeys, EclConfig, ResConfig

from .config_dict_generator import config_dicts, to_config_file


@pytest.mark.usefixtures("setup_tmpdir")
@given(config_dicts())
def test_ecl_config_dict_creates_equal_config(config_dict):
    cwd = os.getcwd()
    filename = config_dict[ConfigKeys.CONFIG_FILE_KEY]
    to_config_file(filename, config_dict)
    res_config = ResConfig(user_config_file=filename)
    config_dict[ConfigKeys.CONFIG_DIRECTORY] = cwd
    assert res_config.ecl_config == EclConfig(config_dict=config_dict)
