import os

import pytest
from hypothesis import given

from ert._c_wrappers.enkf import ConfigKeys, ResConfig

from .config_dict_generator import config_dicts, to_config_file


@pytest.mark.usefixtures("use_tmpdir", "set_site_config")
@given(config_dicts())
def test_site_config_dict_same_as_from_file(config_dict):
    cwd = os.getcwd()
    filename = config_dict[pytest.TEST_CONFIG_FILE_KEY]
    to_config_file(filename, config_dict)
    config_dict[ConfigKeys.CONFIG_DIRECTORY] = cwd
    assert (
        ResConfig(config_dict=config_dict).site_config
        == ResConfig(filename).site_config
    )
