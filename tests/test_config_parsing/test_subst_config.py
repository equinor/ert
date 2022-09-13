import os
import os.path

import pytest
from hypothesis import assume, given

from ert._c_wrappers.enkf import ConfigKeys, ResConfig, SubstConfig

from .config_dict_generator import config_dicts, to_config_file


@pytest.mark.usefixtures("use_tmpdir")
@given(config_dicts())
def test_two_instances_of_same_config_are_equal(config_dict):
    assert SubstConfig(config_dict=config_dict) == SubstConfig(config_dict=config_dict)


@pytest.mark.usefixtures("use_tmpdir")
@given(config_dicts(), config_dicts())
def test_two_instances_of_different_config_are_not_equal(config_dict1, config_dict2):
    assume(config_dict1[ConfigKeys.DEFINE_KEY] != config_dict2[ConfigKeys.DEFINE_KEY])
    assert SubstConfig(config_dict=config_dict1) != SubstConfig(
        config_dict=config_dict2
    )


@pytest.mark.skip(reason="https://github.com/equinor/ert/issues/3802")
@pytest.mark.usefixtures("use_tmpdir")
@given(config_dicts())
def test_from_dict_and_from_file_creates_equal_config(config_dict):
    filename = config_dict[ConfigKeys.CONFIG_FILE_KEY]
    to_config_file(filename, config_dict)
    res_config_from_file = ResConfig(user_config_file=filename)
    subst_config_from_dict = SubstConfig(config_dict=config_dict)
    assert res_config_from_file.subst_config == subst_config_from_dict


@pytest.mark.usefixtures("use_tmpdir")
@given(config_dicts())
def test_complete_config_reads_correct_values(config_dict):
    subst_config = SubstConfig(config_dict=config_dict)
    assert subst_config["<CWD>"] == config_dict[ConfigKeys.CONFIG_DIRECTORY]
    assert subst_config["<CONFIG_PATH>"] == config_dict[ConfigKeys.CONFIG_DIRECTORY]
    for key, value in config_dict[ConfigKeys.DEFINE_KEY].items():
        assert subst_config[key] == value
    for key, value in config_dict[ConfigKeys.DATA_KW_KEY].items():
        assert subst_config[key] == value
    assert subst_config["<RUNPATH_FILE>"] == config_dict[ConfigKeys.RUNPATH_FILE]
    expected_num_cpu = (
        config_dict[ConfigKeys.NUM_CPU] if ConfigKeys.NUM_CPU in config_dict else 1
    )
    assert subst_config["<NUM_CPU>"] == str(expected_num_cpu)


@pytest.mark.usefixtures("use_tmpdir")
@given(config_dicts())
def test_missing_runpath_gives_default_value(config_dict):
    config_dict.pop(ConfigKeys.RUNPATH_FILE)
    subst_config = SubstConfig(config_dict=config_dict)
    assert subst_config["<RUNPATH_FILE>"] == ".ert_runpath_list"


def test_empty_config_raises_error():
    with pytest.raises(ValueError):
        SubstConfig(config_dict={})


@pytest.mark.usefixtures("use_tmpdir")
@given(config_dicts())
def test_missing_config_directory_raises_error(config_dict):
    config_dict.pop(ConfigKeys.CONFIG_DIRECTORY)
    with pytest.raises(ValueError):
        SubstConfig(config_dict=config_dict)


@pytest.mark.usefixtures("use_tmpdir")
@given(config_dicts())
def test_data_file_not_found_raises_error(config_dict):
    config_dict[ConfigKeys.DATA_FILE] = "not_a_file"
    # subst config only tries to read data file if num cpu is not given
    del config_dict[ConfigKeys.NUM_CPU]
    with pytest.raises(IOError):
        SubstConfig(config_dict=config_dict)
