#!/usr/bin/env python
#  Copyright (C) 2011  Statoil ASA, Norway. 
#   
#  The file 'sum_test.py' is part of ERT - Ensemble based Reservoir Tool. 
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
import datetime
import unittest
import time
import ert
import ert.ecl.ecl as ecl
import shutil
import sys
import tempfile
from   ert.util.stringlist import StringList
from   test_util import approx_equal, approx_equalv


base = "ECLIPSE"
path = "test-data/Statoil/ECLIPSE/Gurbat"
case = "%s/%s" % (path , base)



class EclSubmitTest( unittest.TestCase ):

    def setUp(self):
        self.run_path = "%s%s" % (os.getcwd() , tempfile.mkdtemp())
        shutil.copytree( "%s/include" % path , "%s/include" % self.run_path)
        shutil.copy( "%s.DATA" % case , self.run_path ) 

        
    def tearDown(self):
        shutil.rmtree( self.run_path )


    def test_submit(self):
        queue = ecl.EclQueue( max_running = 1 , size = 1)
        job = queue.submit("%s/%s.DATA" % (self.run_path , base))

        while queue.running:
            print "%d" % queue.num_complete , 
            sys.stdout.flush()
            time.sleep( 1 )
        
        sum = ecl.EclSum( "%s/%s" % (self.run_path , base))
        self.assertTrue( isinstance( sum , ecl.EclSum ))
        self.assertEqual( 62 , sum.last_report )


def fast_suite():
    suite = unittest.TestSuite()
    suite.addTest( EclSubmitTest( 'test_submit' ))
    return suite


def test_suite( argv ):
    return fast_suite()


if __name__ == "__main__":
    unittest.TextTestRunner().run( fast_suite() )
