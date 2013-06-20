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
import random
import ert.job_queue.driver as driver
from   test_util import approx_equal, approx_equalv


base = "ECLIPSE_SHORT"
path = "test-data/Statoil/ECLIPSE/Gurbat"
case = "%s/%s" % (path , base)



class EclSubmitTest( unittest.TestCase ):

            
    def make_run_path(self):
        run_path = "%s%s" % (os.getcwd() , "/tmp-%06d" % random.randint(0,999999))
        os.makedirs( run_path )
        shutil.copytree( "%s/include" % path , "%s/include" % run_path)
        shutil.copy( "%s.DATA" % case , run_path ) 
        return run_path
        

    def test_LSF_submit(self):
        num_submit = 6
        queue = ecl.EclQueue( driver_type = driver.LSF_DRIVER , max_running = 4 , size = num_submit)
        path_list = []

        for iens in (range(num_submit)):
            run_path = self.make_run_path()
            path_list.append( run_path )
            job = queue.submit("%s/%s.DATA" % (run_path , base))


        while queue.running:
            time.sleep( 1 )
        

        for path in path_list:
            sum = ecl.EclSum( "%s/%s" % (path , base))
            self.assertTrue( isinstance( sum , ecl.EclSum ))
            self.assertEqual( 2 , sum.last_report )
            
            shutil.rmtree( path )



    def test_LOCAL_submit(self):
        num_submit = 4
        queue = ecl.EclQueue( driver_type = driver.LOCAL_DRIVER , max_running = 2 )
        path_list = []

        for iens in (range(num_submit)):
            run_path = self.make_run_path()
            path_list.append( run_path )
            job = queue.submit("%s/%s.DATA" % (run_path , base))

        queue.submit_complete()
        while queue.running:
            time.sleep( 1 )
        

        for path in path_list:
            sum = ecl.EclSum( "%s/%s" % (path , base))
            self.assertTrue( isinstance( sum , ecl.EclSum ))
            self.assertEqual( 2 , sum.last_report )
            
            shutil.rmtree( path )


def fast_suite():
    suite = unittest.TestSuite()
    suite.addTest( EclSubmitTest( 'test_LSF_submit' ))
    suite.addTest( EclSubmitTest( 'test_LOCAL_submit' ))
    return suite


def test_suite( argv ):
    cwd = os.getcwd()
    work_path = argv[0]
    os.chdir( work_path )
    if not os.path.exists("test-data"):
        os.symlink( "%s/test-data" % cwd , "test-data")
    return fast_suite()


if __name__ == "__main__":
    unittest.TextTestRunner().run( fast_suite() )
