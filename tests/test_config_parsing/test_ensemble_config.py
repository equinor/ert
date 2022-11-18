import pytest
from hypothesis import assume, given

from ert._c_wrappers.config import ConfigValidationError
from ert._c_wrappers.enkf import ConfigKeys, ResConfig

from .config_dict_generator import config_dicts, to_config_file


@pytest.mark.usefixtures("use_tmpdir")
@given(config_dicts())
def test_ensemble_config_from_file_and_dict_coincide(config_dict):
    filename = "config.ert"
    to_config_file(filename, config_dict)

    res_config_from_file = ResConfig(user_config_file=filename)
    res_config_from_dict = ResConfig(config_dict=config_dict)

    assert res_config_from_file.ensemble_config == res_config_from_dict.ensemble_config


@pytest.mark.skip(reason="github.com/equinor/ert/issues/4070")
@pytest.mark.usefixtures("use_tmpdir")
@given(config_dicts())
def test_ensemble_config_errors_on_unknown_function_in_field(config_dict):
    filename = "config.ert"
    assume(
        ConfigKeys.FIELD_KEY in config_dict
        and len(config_dict[ConfigKeys.FIELD_KEY]) > 0
    )
    silly_function_name = "NORMALIZE_EGGS"
    config_dict[ConfigKeys.FIELD_KEY][ConfigKeys.INIT_TRANSFORM] = silly_function_name
    to_config_file(filename, config_dict)
    with pytest.raises(
        expected_exception=ValueError, match=f"unknown function.*{silly_function_name}"
    ):
        ResConfig(user_config_file=filename)


@pytest.mark.skip(reason="github.com/equinor/ert/issues/4071")
@pytest.mark.usefixtures("use_tmpdir")
@given(config_dicts())
def test_ensemble_config_errors_on_identical_name_for_two_enkf_config_nodes(
    config_dict,
):
    filename = "config.ert"
    two_keys = [ConfigKeys.FIELD_KEY, ConfigKeys.GEN_DATA]
    for key in two_keys:
        assume(key in config_dict and len(config_dict[key]) > 0)
    one_node = config_dict[two_keys[0]][0]
    other_node = config_dict[two_keys[1]][0]
    one_node[ConfigKeys.NAME] = other_node[ConfigKeys.NAME]

    to_config_file(filename, config_dict)
    err_msg_match = "duplicate key"
    with pytest.raises(
        expected_exception=ConfigValidationError,
        match=err_msg_match,
    ):
        ResConfig(user_config_file=filename)
    with pytest.raises(
        expected_exception=ConfigValidationError,
        match=err_msg_match,
    ):
        ResConfig(config_dict=config_dict)
