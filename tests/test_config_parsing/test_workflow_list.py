import pytest
from hypothesis import given

from ert._c_wrappers.enkf import ErtWorkflowList, ResConfig

from .config_dict_generator import config_generators


@pytest.mark.skip(reason="https://github.com/equinor/ert/issues/2536")
@given(config_generators())
def test_ert_workflow_list_dict_creates_equal_config(
    tmp_path_factory, config_generator
):
    filename = "config.ert"
    with config_generator(tmp_path_factory, filename) as config_dict:
        res_config = ResConfig(user_config_file=filename)
        assert res_config.ert_workflow_list == ErtWorkflowList.from_dict(config_dict)
