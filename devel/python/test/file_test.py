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

import filecmp
import datetime
import unittest
import shutil
import time
import os
import os.path
import ert
import ert.ecl.ecl as ecl
from   test_util import approx_equal, approx_equalv, file_equal
from   ert.util.test_area import TestArea

file     = "test-data/Statoil/ECLIPSE/Gurbat/ECLIPSE.UNRST"
fmt_file = "test-data/Statoil/ECLIPSE/Gurbat/ECLIPSE.FUNRST"

test_data_root = os.path.abspath( os.path.join( os.path.dirname( os.path.abspath( __file__)) , "../../"))

def test_path( path ):
    return os.path.join( test_data_root , path )


def load_missing():
    ecl.EclFile( "No/Does/not/exist")
    

class FileTest( unittest.TestCase ):

    
    def testIOError(self):
        self.assertRaises( IOError , load_missing)

    
    def test_fwrite( self ):
        work_area = TestArea("python/ecl_file/fwrite")
        rst_file = ecl.EclFile( test_path(file) )
        fortio = ecl.FortIO.writer("ECLIPSE.UNRST")
        rst_file.fwrite( fortio )
        fortio.close()
        rst_file.close()
        self.assertTrue( file_equal( "ECLIPSE.UNRST" , test_path(file)) ) 


    def test_save(self):
        work_area = TestArea("python/ecl_file/save")
        work_area.copy_file( test_path( file ))
        rst_file = ecl.EclFile( "ECLIPSE.UNRST" , flags = ecl.ECL_FILE_WRITABLE )
        swat0 = rst_file["SWAT"][0]
        swat0.assign( 0.75 )
        rst_file.save_kw( swat0 )
        rst_file.close( )
        self.assertFalse( file_equal( "ECLIPSE.UNRST" , test_path(file)) )
        
        rst_file1 = ecl.EclFile( test_path(file) )
        rst_file2 = ecl.EclFile( "ECLIPSE.UNRST" , flags = ecl.ECL_FILE_WRITABLE)
        
        swat1 = rst_file1["SWAT"][0]
        swat2 = rst_file2["SWAT"][0]
        swat2.assign( swat1 )

        rst_file2.save_kw( swat2 )
        self.assertTrue( swat1.equal( swat2 ))
        rst_file1.close()
        rst_file2.close()

        # Random failure ....
        self.assertTrue( file_equal( "ECLIPSE.UNRST" , test_path(file)) ) 
        

    def test_save_fmt(self):
        work_area = TestArea("python/ecl_file/save_fmt")
        work_area.copy_file( test_path( test_path(fmt_file )))
        rst_file = ecl.EclFile( "ECLIPSE.FUNRST" , flags = ecl.ECL_FILE_WRITABLE)
        swat0 = rst_file["SWAT"][0]
        swat0.assign( 0.75 )
        rst_file.save_kw( swat0 )
        rst_file.close( )
        self.assertFalse( file_equal( "ECLIPSE.FUNRST" , fmt_file ) )
        
        rst_file1 = ecl.EclFile( fmt_file )
        rst_file2 = ecl.EclFile( "ECLIPSE.FUNRST" , flags = ecl.ECL_FILE_WRITABLE)
        
        swat1 = rst_file1["SWAT"][0]
        swat2 = rst_file2["SWAT"][0]

        swat2.assign( swat1 )
        rst_file2.save_kw( swat2 )
        self.assertTrue( swat1.equal( swat2 ))
        rst_file1.close()
        rst_file2.close()

        # Random failure ....
        self.assertTrue( file_equal( "ECLIPSE.FUNRST" , fmt_file ) ) 
        


def slow_suite():
    suite = unittest.TestSuite()
    suite.addTest( FileTest( 'test_save' ))
    suite.addTest( FileTest( 'test_save_fmt' ))
    return suite


def fast_suite():
    suite = unittest.TestSuite()
    suite.addTest( FileTest( 'testIOError' )) 
    suite.addTest( FileTest( 'test_fwrite' ))
    return suite




def test_suite( argv ):
    test_list = fast_suite()
    if argv:
        if argv[0][0] == "T":
            for t in slow_suite():
                test_list.addTest( t )
    return test_list

            

if __name__ == "__main__":
    unittest.TextTestRunner().run( fast_suite() )
    unittest.TextTestRunner().run( slow_suite() )
