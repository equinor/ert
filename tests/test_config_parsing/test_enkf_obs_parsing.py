import pytest
from hypothesis import given

from ert._c_wrappers.enkf import EnkfObs, ErtConfig

from .config_dict_generator import config_generators


@pytest.mark.usefixtures("set_site_config")
@given(config_generators())
def test_that_creating_enkf_obs_from_generated_values_does_not_produce_errors(
    tmp_path_factory, config_generator
):
    filename = "config.ert"
    with config_generator(tmp_path_factory, filename):
        observations = EnkfObs.from_ert_config(ErtConfig.from_file(filename))
        assert observations.error == ""
