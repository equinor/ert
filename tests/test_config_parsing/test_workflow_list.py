import pytest
from hypothesis import given

from ert._c_wrappers.enkf import ErtWorkflowList, ResConfig

from .config_dict_generator import config_dicts, to_config_file


@pytest.mark.skip(reason="https://github.com/equinor/ert/issues/2536")
@pytest.mark.usefixtures("use_tmpdir")
@given(config_dicts())
def test_ert_workflow_list_dict_creates_equal_config(config_dict):
    filename = "config.ert"
    to_config_file(filename, config_dict)
    res_config = ResConfig(user_config_file=filename)
    assert res_config.ert_workflow_list == ErtWorkflowList.from_dict(config_dict)
