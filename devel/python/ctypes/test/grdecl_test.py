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

import datetime
import unittest
import ert
import ert.ecl.ecl as ecl
from   test_util import approx_equal, approx_equalv, file_equal


src_file = "data/eclipse/case/include/example_permx.GRDECL"


class GRDECLTest( unittest.TestCase ):

    def setUp(self):
        pass


    def testLoad( self ):
        kw = ecl.EclKW.read_grdecl( open( src_file , "r") , "PERMX")
        self.assertTrue( kw )


    def testReLoad( self ):
        kw = ecl.EclKW.read_grdecl( open( src_file , "r") , "PERMX")
        tmp_file1 = "/tmp/permx1.grdecl"
        tmp_file2 = "/tmp/permx2.grdecl"

        fileH = open( tmp_file1 , "w")
        kw.write_grdecl( fileH )
        fileH.close()

        kw1 = ecl.EclKW.read_grdecl( open( tmp_file1 , "r") , "PERMX") 
        
        fileH = open( tmp_file2 , "w")
        kw1.write_grdecl( fileH )
        fileH.close()

        self.assertTrue( file_equal( tmp_file1 , tmp_file2 ))


def fast_suite():
    suite = unittest.TestSuite()
    suite.addTest( GRDECLTest( 'testLoad' ))
    suite.addTest( GRDECLTest( 'testReLoad' ))
    return suite
