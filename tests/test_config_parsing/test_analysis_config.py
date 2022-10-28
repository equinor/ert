import os

import pytest
from hypothesis import given

from ert._c_wrappers.enkf import AnalysisConfig, ConfigKeys, ResConfig

from .config_dict_generator import config_dicts, to_config_file


@pytest.mark.usefixtures("use_tmpdir")
@given(config_dicts())
def test_analysis_config_config_same_as_from_file(config_dict):
    cwd = os.getcwd()
    filename = config_dict[pytest.TEST_CONFIG_FILE_KEY]
    to_config_file(filename, config_dict)
    config_dict[ConfigKeys.CONFIG_DIRECTORY] = cwd

    analysis_config_from_file = ResConfig(filename).analysis_config
    analysis_config_from_dict = AnalysisConfig.from_dict(config_dict=config_dict)
    assert analysis_config_from_file == analysis_config_from_dict
