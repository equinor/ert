import os
import os.path

import pytest
from hypothesis import assume, given

from ert._c_wrappers.enkf import ConfigKeys, ResConfig

from .config_dict_generator import config_dicts, to_config_file


@pytest.mark.usefixtures("use_tmpdir")
@given(config_dicts(), config_dicts())
def test_different_defines_give_different_subst_lists(config_dict1, config_dict2):
    assume(config_dict1[ConfigKeys.DEFINE_KEY] != config_dict2[ConfigKeys.DEFINE_KEY])
    assert (
        ResConfig(config_dict=config_dict1).substitution_list
        != ResConfig(config_dict=config_dict2).substitution_list
    )


@pytest.mark.usefixtures("use_tmpdir")
@given(config_dicts())
def test_from_dict_and_from_file_creates_equal_subst_lists(config_dict):
    filename = dict(config_dict[ConfigKeys.DEFINE_KEY])["<CONFIG_FILE>"]
    to_config_file(filename, config_dict)
    res_config_from_file = ResConfig(user_config_file=filename)
    res_config_from_dict = ResConfig(config_dict=config_dict)
    assert (
        res_config_from_file.substitution_list == res_config_from_dict.substitution_list
    )


@pytest.mark.usefixtures("use_tmpdir")
@given(config_dicts())
def test_complete_config_reads_correct_values(config_dict):
    filename = dict(config_dict[ConfigKeys.DEFINE_KEY])["<CONFIG_FILE>"]
    substitution_list = ResConfig(config_dict=config_dict).substitution_list
    cwd = os.getcwd()
    assert substitution_list["<CWD>"] == cwd
    assert substitution_list["<CONFIG_PATH>"] == cwd
    assert substitution_list["<CONFIG_FILE>"] == filename
    assert substitution_list["<CONFIG_FILE_BASE>"] == os.path.splitext(filename)[0]
    for key, value in dict(config_dict[ConfigKeys.DEFINE_KEY]).items():
        assert substitution_list[key] == value
    for key, value in dict(config_dict[ConfigKeys.DATA_KW_KEY]).items():
        assert substitution_list[key] == value
    cwd = os.getcwd()
    assert substitution_list["<RUNPATH_FILE>"] == os.path.join(
        cwd, config_dict[ConfigKeys.RUNPATH_FILE]
    )
    expected_num_cpu = (
        config_dict[ConfigKeys.NUM_CPU] if ConfigKeys.NUM_CPU in config_dict else 1
    )
    assert substitution_list["<NUM_CPU>"] == str(expected_num_cpu)


@pytest.mark.usefixtures("use_tmpdir")
@given(config_dicts())
def test_missing_runpath_gives_default_value(config_dict):
    config_dict.pop(ConfigKeys.RUNPATH_FILE)
    res_config = ResConfig(config_dict=config_dict)
    expected_runpath_filepath = os.path.join(os.getcwd(), ".ert_runpath_list")
    assert res_config.substitution_list["<RUNPATH_FILE>"] == expected_runpath_filepath
