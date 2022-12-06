import hypothesis.strategies as st
import pytest
from hypothesis import given

from ert._c_wrappers.enkf import ResConfig

from .config_dict_generator import config_dicts, to_config_file


@pytest.mark.usefixtures("use_tmpdir", "set_site_config")
@given(config_dicts())
def test_queue_config_dict_same_as_from_file(config_dict):
    filename = "config.ert"
    to_config_file(filename, config_dict)
    assert (
        ResConfig(filename).queue_config
        == ResConfig(config_dict=config_dict).queue_config
    )


@pytest.mark.usefixtures("use_tmpdir", "set_site_config")
@given(st.integers(min_value=1, max_value=300))
def test_queue_config_default_max_running_is_unlimited(num_real):
    filename = "config.ert"
    with open(filename, "w") as f:
        f.write(f"NUM_REALIZATIONS {num_real}\nQUEUE_SYSTEM SLURM\n")
    # get_max_running() == 0 means unlimited
    assert ResConfig(filename).queue_config.create_job_queue().get_max_running() == 0
