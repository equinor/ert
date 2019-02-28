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

from tests import ResTest
from res.test import ErtTestContext



class DataKWTest(ResTest):

    def test_it(self):
        config = self.createTestPath("local/custom_kw/mini_config_define")
        with ErtTestContext("mini_config_define", config) as context:
            ert = context.getErt()
            data_kw = ert.getDataKW( )
            my_path = data_kw["MY_PATH"]
            self.assertEqual( my_path , os.getcwd())
