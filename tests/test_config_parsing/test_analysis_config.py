import os

import pytest
from hypothesis import given

from ert._c_wrappers.enkf import AnalysisConfig, ConfigKeys

from .config_dict_generator import config_dicts, to_config_file


@pytest.mark.skip(reason="https://github.com/equinor/ert/issues/2560")
@pytest.mark.usefixtures("setup_tmpdir")
@given(config_dicts())
def test_analysis_config_config_same_as_from_file(config_dict):
    cwd = os.getcwd()
    filename = config_dict[ConfigKeys.CONFIG_FILE_KEY]
    to_config_file(filename, config_dict)
    config_dict[ConfigKeys.CONFIG_DIRECTORY] = cwd
    assert AnalysisConfig(filename) == AnalysisConfig(config_dict=config_dict)
