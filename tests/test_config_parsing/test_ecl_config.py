import os
import os.path

import pytest
from hypothesis import given

from ert._c_wrappers.enkf import ConfigKeys, EclConfig, ResConfig

from .config_dict_generator import config_dicts, to_config_file


@pytest.mark.usefixtures("use_tmpdir")
@given(config_dicts())
def test_ecl_config_dict_creates_equal_config(config_dict):
    cwd = os.getcwd()
    filename = config_dict[ConfigKeys.CONFIG_FILE_KEY]
    to_config_file(filename, config_dict)
    res_config = ResConfig(user_config_file=filename)
    config_dict[ConfigKeys.CONFIG_DIRECTORY] = cwd
    assert res_config.ecl_config == EclConfig.from_dict(config_dict=config_dict)


@pytest.mark.usefixtures("use_tmpdir")
def test_ecl_config_fails_on_non_sensical_refcase_file():
    refcase_file = "CEST_PAS_UNE_REFCASE"
    refcase_file_content = """
_________________________________________     _____    ____________________
\\______   \\_   _____/\\_   _____/\\_   ___ \\   /  _  \\  /   _____/\\_   _____/
 |       _/|    __)_  |    __)  /    \\  \\/  /  /_\\  \\ \\_____  \\  |    __)_
 |    |   \\|        \\ |     \\   \\     \\____/    |    \\/        \\ |        \\
 |____|_  /_______  / \\___  /    \\______  /\\____|__  /_______  //_______  /
        \\/        \\/      \\/            \\/         \\/        \\/         \\/
"""
    with open(refcase_file, "w+", encoding="utf-8") as refcase_file_handler:
        refcase_file_handler.write(refcase_file_content)
    with pytest.raises(expected_exception=IOError, match=refcase_file):
        EclConfig(refcase_file=refcase_file)


@pytest.mark.usefixtures("use_tmpdir")
def test_ecl_config_fails_on_non_sensical_grid_file():
    grid_file = "BRICKWALL"
    # NB: this is just silly ASCII content, not even close to a correct GRID file
    grid_file_content = """
_|___|___|___|___|___|___|___|___|___|___|___|___|___|___|___|___|___|
___|___|___|___|___|___|___|___|___|___|___|___|___|___|___|___|___|__
_|___|___|___|___|___|___|___|___|___|___|___|___|___|___|___|___|___|
___|___|___|___|___|___|___|___|___|___|___|___|___|___|___|___|___|__
_|___|___|___|___|___|___|___|___|___|___|___|___|___|___|___|___|___|
"""
    with open(grid_file, "w+", encoding="utf-8") as grid_file_handler:
        grid_file_handler.write(grid_file_content)
    with pytest.raises(expected_exception=ValueError, match=grid_file):
        EclConfig(grid_file=grid_file)
