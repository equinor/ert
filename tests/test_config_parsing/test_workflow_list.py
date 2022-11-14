import os

import pytest
from hypothesis import given

from ert._c_wrappers.enkf import ConfigKeys, ErtWorkflowList, ResConfig

from .config_dict_generator import config_dicts, to_config_file


@pytest.mark.skip(reason="https://github.com/equinor/ert/issues/2536")
@pytest.mark.usefixtures("use_tmpdir")
@given(config_dicts())
def test_ert_workflow_list_dict_creates_equal_config(config_dict):
    cwd = os.getcwd()
    filename = config_dict[pytest.TEST_CONFIG_FILE_KEY]
    to_config_file(filename, config_dict)
    res_config = ResConfig(user_config_file=filename)
    config_dict[ConfigKeys.CONFIG_DIRECTORY] = cwd
    assert res_config.ert_workflow_list == ErtWorkflowList(
        config_dict=config_dict,
    )
