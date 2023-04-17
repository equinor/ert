import pytest
from hypothesis import given

from ert._c_wrappers.enkf import EnkfObs, ErtConfig

from .config_dict_generator import config_generators


@pytest.mark.usefixtures("set_site_config")
@given(config_generators())
def test_that_enkf_obs_keys_are_ordered(tmp_path_factory, config_generator):
    filename = "config.ert"
    with config_generator(tmp_path_factory, filename) as config_values:
        ert_config = ErtConfig.from_file(filename)
        observations = EnkfObs.from_ert_config(ert_config)
        assert observations.error == ""
        for o in config_values.observations:
            assert observations.hasKey(o.name)
        assert sorted(set(o.name for o in config_values.observations)) == list(
            observations.getMatchingKeys("*")
        )
