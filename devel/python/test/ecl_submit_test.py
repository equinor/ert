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
import getpass
import datetime
import unittest
import time
import shutil
import ert
import ert.ecl.ecl as ecl
import shutil
import sys
import random
import ert.job_queue.driver as driver
from   ert.util.test_area import TestArea
from   test_util import approx_equal, approx_equalv

test_data_root = os.path.abspath( os.path.join( os.path.dirname( os.path.abspath( __file__)) , "../../"))

base = "ECLIPSE_SHORT"
path = "test-data/Statoil/ECLIPSE/Gurbat"
case = "%s/%s" % (path , base)
nfs_work_path = "/scratch/ert"

def test_path( path ):
    return os.path.join( test_data_root , path )




class EclSubmitTest( unittest.TestCase ):
    

    def make_run_path(self , iens , prefix = None):
        run_path = "run%d" % iens
        if os.path.exists( run_path ):
            shutil.rmtree( run_path )

        os.makedirs( run_path )
        shutil.copytree( "%s/include" % test_path( path ) , "%s/include" % run_path)
        shutil.copy( "%s.DATA" % test_path(case) , run_path ) 
        return os.path.abspath( run_path )
        

    def test_LSF_submit(self):
        root = os.path.join(nfs_work_path , getpass.getuser() , "ert-test/python/ecl_submit/LSF" )
        if not os.path.exists( root ):
            os.makedirs( root )
        os.chdir( root )
        print root

        num_submit = 6
        queue = ecl.EclQueue( driver_type = driver.LSF_DRIVER , max_running = 4 , size = num_submit)
        path_list = []

        for iens in (range(num_submit)):
            run_path = self.make_run_path( iens )
            path_list.append( run_path )
            job = queue.submit("%s/%s.DATA" % (run_path , base))


        while queue.running:
            time.sleep( 1 )
        

        for path in path_list:
            sum = ecl.EclSum( "%s/%s" % (path , base))
            self.assertTrue( isinstance( sum , ecl.EclSum ))
            self.assertEqual( 2 , sum.last_report )
            



    def test_LOCAL_submit(self):
        work_area = TestArea( "python/ecl_submit/LOCAL" , True )
        num_submit = 4
        queue = ecl.EclQueue( driver_type = driver.LOCAL_DRIVER , max_running = 2 )
        path_list = []
        

        for iens in (range(num_submit)):
            run_path = self.make_run_path(iens)
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
    suite = unittest.TestSuite()
    queue_name = argv[0]
    if queue_name == "LSF":
        suite.addTest( EclSubmitTest( 'test_LSF_submit' ))
        nfs_work_path = argv[1]
    elif queue_name == "LOCAL":
        suite.addTest( EclSubmitTest( 'test_LOCAL_submit' ))
    return suite

if __name__ == "__main__":
    unittest.TextTestRunner().run( fast_suite() )
