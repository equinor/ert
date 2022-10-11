import os
import os.path

import pytest
from hypothesis import given, settings, reproduce_failure

from ert._c_wrappers.enkf import ConfigKeys, EnsembleConfig, ResConfig

from .config_dict_generator import config_dicts, to_config_file


@pytest.mark.usefixtures("use_tmpdir")
@settings(print_blob=True)
@given(config_dicts())
def test_ensemble_config_works_without_grid(config_dict):
    cwd = os.getcwd()
    filename = config_dict[ConfigKeys.CONFIG_FILE_KEY]
    # config_dict.pop(ConfigKeys.GRID)
    to_config_file(filename, config_dict)
    config_dict[ConfigKeys.CONFIG_DIRECTORY] = cwd

    res_config_from_file = ResConfig(user_config_file=filename)
    res_config_from_dict = ResConfig(config_dict=config_dict)

    with open(filename, "r+") as file_handler:
        print(file_handler.read())

    print(config_dict[ConfigKeys.FIELD_KEY])
    print(res_config_from_file.ensemble_config)
    assert res_config_from_file.ensemble_config == res_config_from_dict.ensemble_config
    # assert False


@pytest.mark.skip(reason="github.com/equinor/ert/issues/4070")
@pytest.mark.usefixtures("use_tmpdir")
@given(config_dicts())
def test_ensemble_config_errors_on_unknown_function_in_field(config_dict):
    filename = config_dict[ConfigKeys.CONFIG_FILE_KEY]
    if ConfigKeys.FIELD_KEY not in config_dict:
        return
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
def test_ensemble_config_errors_on_double_field(config_dict):
    # TODO in the logs i see a doubled OUT_FILE, not two fields with equal
    # name, as i first suspected - figure out what's going on here!
    filename = config_dict[ConfigKeys.CONFIG_FILE_KEY]
    if ConfigKeys.FIELD_KEY not in config_dict:
        return
    # TODO manipulate FIELD to be degenerate in the desired way
    to_config_file(filename, config_dict)
    with pytest.raises(
        expected_exception=ValueError, match=f"TODO this should be a meaningful message"
    ):
        ResConfig(user_config_file=filename)


@pytest.mark.skip(reason="github.com/equinor/ert/issues/4072")
@pytest.mark.usefixtures("use_tmpdir")
@given(config_dicts())
def test_ensemble_config_gen_data_report_steps_and_gen_obs_restart_dependency(
    config_dict,
):
    pass
