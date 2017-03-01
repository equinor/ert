#  Copyright (C) 2015  Statoil ASA, Norway. 
#   
#  The file 'test_testarea.py' is part of ERT - Ensemble based Reservoir Tool.
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

from ert.test import ExtendedTestCase, TestAreaContext

class TestTestArea(ExtendedTestCase):
    
    def test_test_area_ok(self):
        orig_cwd = os.getcwd()
        f_name = "this_is_not_a_file?"
        self.assertFalse(os.path.isfile(f_name))

        with TestAreaContext("give_me_a_tmp_dir"):
            self.assertNotEqual(orig_cwd, os.getcwd())

            f_name = os.path.join(os.getcwd(), f_name)
            with open(f_name, "w") as f:
                f.write("Some text")
            self.assertTrue(os.path.isfile(f_name))

        self.assertFalse(os.path.isfile(f_name))
