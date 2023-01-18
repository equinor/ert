import pytest
from hypothesis import assume, given

from ert._c_wrappers.enkf import ConfigKeys, ResConfig

from .config_dict_generator import config_generators, to_config_file


@pytest.mark.skip(reason="github.com/equinor/ert/issues/4070")
@given(config_generators())
def test_ensemble_config_errors_on_unknown_function_in_field(
    tmp_path_factory, config_generator
):
    filename = "config.ert"
    with config_generator(tmp_path_factory, filename) as config_dict:
        assume(
            ConfigKeys.FIELD_KEY in config_dict
            and len(config_dict[ConfigKeys.FIELD_KEY]) > 0
        )
        silly_function_name = "NORMALIZE_EGGS"
        config_dict[ConfigKeys.FIELD_KEY][
            ConfigKeys.INIT_TRANSFORM
        ] = silly_function_name
        to_config_file(filename, config_dict)
        with pytest.raises(
            expected_exception=ValueError,
            match=f"unknown function.*{silly_function_name}",
        ):
            ResConfig(user_config_file=filename)
