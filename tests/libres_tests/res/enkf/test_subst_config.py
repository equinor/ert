#  Copyright (C) 2017  Equinor ASA, Norway.
#
#  The file 'test_subst_config.py' is part of ERT - Ensemble based Reservoir Tool.
#
#  ERT is free software: you can redistribute it and/or modify
#  it under the terms of the GNU General Public License as published by
#  the Free Software Foundation, either version 3 of the License, or
#  (at your option) any later version.
#
#  ERT is distributed in the hope that it will be useful, but WITHOUT ANY
#  WARRANTY; without even the implied warranty of MERCHANTABILITY or
#  FITNESS FOR A PARTICULAR PURPOSE.
#
#  See the GNU General Public License at <http://www.gnu.org/licenses/gpl.html>
#  for more details.
import os
import os.path

import pytest
from hypothesis import assume, given
from res.enkf import ConfigKeys, ResConfig, SubstConfig

from config_dict_generator import config_dicts, to_config_file


@pytest.mark.usefixtures("setup_tmpdir")
@given(config_dicts())
def test_two_instances_of_same_config_are_equal(config_dict):
    assert SubstConfig(config_dict=config_dict) == SubstConfig(config_dict=config_dict)


@pytest.mark.usefixtures("setup_tmpdir")
@given(config_dicts(), config_dicts())
def test_two_instances_of_different_config_are_not_equal(config_dict1, config_dict2):
    assume(config_dict1[ConfigKeys.DEFINE_KEY] != config_dict2[ConfigKeys.DEFINE_KEY])
    assert SubstConfig(config_dict=config_dict1) != SubstConfig(
        config_dict=config_dict2
    )


@pytest.mark.usefixtures("setup_tmpdir")
@given(config_dicts())
def test_old_and_new_constructor_creates_equal_config(config_dict):
    cwd = os.getcwd()
    filename = config_dict[ConfigKeys.CONFIG_FILE_KEY]
    to_config_file(filename, config_dict)
    res_config = ResConfig(user_config_file=filename)
    config_dict[ConfigKeys.CONFIG_DIRECTORY] = cwd
    assert res_config.subst_config == SubstConfig(config_dict=config_dict)


@pytest.mark.usefixtures("setup_tmpdir")
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
    assert subst_config["<NUM_CPU>"] == "1"


@pytest.mark.usefixtures("setup_tmpdir")
@given(config_dicts())
def test_missing_runpath_gives_default_value(config_dict):
    config_dict.pop(ConfigKeys.RUNPATH_FILE)
    subst_config = SubstConfig(config_dict=config_dict)
    assert subst_config["<RUNPATH_FILE>"] == ".ert_runpath_list"


def test_empty_config_raises_error():
    with pytest.raises(ValueError):
        SubstConfig(config_dict={})


@pytest.mark.usefixtures("setup_tmpdir")
@given(config_dicts())
def test_missing_config_directory_raises_error(config_dict):
    config_dict.pop(ConfigKeys.CONFIG_DIRECTORY)
    with pytest.raises(ValueError):
        SubstConfig(config_dict=config_dict)


@pytest.mark.usefixtures("setup_tmpdir")
@given(config_dicts())
def test_data_file_not_found_raises_error(config_dict):
    config_dict[ConfigKeys.DATA_FILE] = "not_a_file"
    with pytest.raises(IOError):
        SubstConfig(config_dict=config_dict)
