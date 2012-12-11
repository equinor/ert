#!/usr/bin/env python
#  Copyright (C) 2012  Statoil ASA, Norway. 
#   
#  The file 'config_test.py' is part of ERT - Ensemble based Reservoir Tool. 
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
import unittest
import stat
import math
import ert
import ert.ecl.ecl as ecl
import ert.config.config as config

import sys
from   test_util import *



class ConfigTest( unittest.TestCase ):
    
    def setUp( self ):
        self.file_list = []


    def test_parse(self):
        conf = config.ConfigParser()
        conf.add("RSH_HOST" , False)
        conf.parse("test-data/local/config/simple_config" , unrecognized = config.CONFIG_UNRECOGNIZED_IGNORE)
        self.assertTrue( True )



def fast_suite():
    suite = unittest.TestSuite()
    suite.addTest( ConfigTest( 'test_parse' ))
    return suite

                   

if __name__ == "__main__":
    unittest.TextTestRunner().run( fast_suite() )


