#  Copyright (C) 2017  Equinor ASA, Norway.
#
#  The file 'test_rng_config.py' is part of ERT - Ensemble based Reservoir Tool.
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

import pytest
from config_dict_generator import config_dicts, to_config_file
from hypothesis import given
from res.enkf import ConfigKeys, ResConfig, RNGConfig


@pytest.mark.usefixtures("setup_tmpdir")
@given(config_dicts())
def test_site_config_config_same_as_from_file(config_dict):
    cwd = os.getcwd()
    filename = config_dict[ConfigKeys.CONFIG_FILE_KEY]
    to_config_file(filename, config_dict)
    config_dict[ConfigKeys.CONFIG_DIRECTORY] = cwd
    assert ResConfig(filename).rng_config == RNGConfig(config_dict=config_dict)
