import pytest
from hypothesis import given

from ert._c_wrappers.enkf import ResConfig

from .config_dict_generator import config_dicts, to_config_file


@pytest.mark.usefixtures("use_tmpdir", "set_site_config")
@given(config_dicts())
def test_site_config_dict_same_as_from_file(config_dict):
    filename = "config.ert"
    to_config_file(filename, config_dict)
    assert (
        ResConfig(config_dict=config_dict).site_config
        == ResConfig(filename).site_config
    )
