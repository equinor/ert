#  Copyright (C) 2017  Equinor ASA, Norway.
#
#  The file 'test_data_kw_define.py' is part of ERT - Ensemble based Reservoir Tool.
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

from res.enkf import ResConfig, EnKFMain


@pytest.mark.usefixtures("use_tmpdir")
def test_data_kw():
    # Write a minimal config file with DEFINE
    with open("config_file.ert", "w") as fout:
        fout.write("NUM_REALIZATIONS 1\nDEFINE MY_PATH <CONFIG_PATH>")
    res_config = ResConfig("config_file.ert")
    ert = EnKFMain(res_config)
    data_kw = ert.getDataKW()
    my_path = data_kw["MY_PATH"]
    assert my_path == os.getcwd()
