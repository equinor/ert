import pytest

from ert._c_wrappers.enkf import SummaryConfig


@pytest.mark.usefixtures("use_tmpdir")
def test_summary_config():
    summary_config = SummaryConfig(key="ALT1")
    obs_list = ["ABC", "GHI", "DEF"]
    assert summary_config.get_observation_keys() == []
    summary_config.update_observation_keys(obs_list)
    assert summary_config.get_observation_keys() == sorted(obs_list)
