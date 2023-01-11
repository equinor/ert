import pytest
from hypothesis import assume, given

from ert._c_wrappers.config import ConfigValidationError
from ert._c_wrappers.enkf import ConfigKeys, ResConfig

from .config_dict_generator import config_generators, to_config_file


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
        fieldlist = list(config_dict[ConfigKeys.FIELD_KEY][0])
        alteredfieldlist = []

        for val in fieldlist:
            if "INIT_TRANSFORM" in val:
                alteredfieldlist.append("INIT_TRANSFORM:" + silly_function_name)
            else:
                alteredfieldlist.append(val)

        mylist = []
        mylist.append(tuple(alteredfieldlist))

        config_dict[ConfigKeys.FIELD_KEY] = mylist

        to_config_file(filename, config_dict)
        with pytest.raises(
            expected_exception=ValueError,
            match=f"FIELD INIT_TRANSFORM:{silly_function_name} is an invalid function",
        ):
            ResConfig(user_config_file=filename)


@pytest.mark.skip(reason="github.com/equinor/ert/issues/4071")
@given(config_generators())
def test_ensemble_config_errors_on_identical_name_for_two_enkf_config_nodes(
    tmp_path_factory,
    config_generator,
):
    filename = "config.ert"
    with config_generator(tmp_path_factory, filename) as config_dict:
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
