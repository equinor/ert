#!/usr/bin/env python
#  Copyright (C) 2011  Statoil ASA, Norway. 
#   
#  The file 'grid_test.py' is part of ERT - Ensemble based Reservoir Tool. 
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

import sys
import ert
import ert.ecl.ecl as ecl
import datetime
import time
import os
import os.path
import unittest
import ert
import ert.ecl.ecl as ecl
from   ert.util.tvector import DoubleVector
from   ert.util.tvector import DoubleVector
from   ert.util.test_area import TestArea
from   numpy import isnan
from   test_util import approx_equal, approx_equalv
test_data_root = os.path.abspath( os.path.join( os.path.dirname( os.path.abspath( __file__)) , "../../"))

def test_path( path ):
    return os.path.join( test_data_root , path )

def egrid_file():
    return test_path( "test-data/Statoil/ECLIPSE/Gurbat/ECLIPSE.EGRID" )

def grid_file():
    return test_path( "test-data/Statoil/ECLIPSE/Gurbat/ECLIPSE.GRID")

def grdecl_file():
    return test_path( "test-data/Statoil/ECLIPSE/Gurbat/include/example_grid_sim.GRDECL")

def load_does_not_exist():
    g = ecl.EclGrid( "/does/not/exist.EGRID" )
    

class GridTest( unittest.TestCase ):

    def testGRID( self ):
        grid = ecl.EclGrid( grid_file() )
        self.assertTrue( grid )


    def testEGRID( self ):
        grid = ecl.EclGrid( egrid_file() )
        self.assertTrue( grid )


    def create(self , filename , load_actnum = True):
        fileH = open( filename , "r")
        specgrid = ecl.EclKW.read_grdecl( fileH , "SPECGRID" , ecl_type = ecl.ECL_INT_TYPE , strict = False)
        zcorn    = ecl.EclKW.read_grdecl( fileH , "ZCORN" )
        coord    = ecl.EclKW.read_grdecl( fileH , "COORD" )
        if load_actnum:
            actnum   = ecl.EclKW.read_grdecl( fileH , "ACTNUM" , ecl_type = ecl.ECL_INT_TYPE )
        else:
            actnum   = None
            
        mapaxes  = ecl.EclKW.read_grdecl( fileH , "MAPAXES" )
        grid = ecl.EclGrid.create( specgrid , zcorn , coord , actnum , mapaxes = mapaxes)
        return grid


    def testRect(self):
        work_area = TestArea("python/grid-test/testRect" , True)
        a1 = 1.0
        a2 = 2.0
        a3 = 3.0
        grid = ecl.EclGrid.create_rectangular((9,9,9) , (a1,a2,a3))
        grid.save_EGRID( "rect.EGRID" )
        grid2 = ecl.EclGrid( "rect.EGRID")
        self.assertTrue( grid )
        self.assertTrue( grid2 )
        
        (x,y,z) = grid.get_xyz( ijk=(4,4,4) )
        self.assertTrue( approx_equalv( [x,y,z],[4.5 * a1,4.5*a2,4.5*a3] ))

        v = grid.cell_volume( ijk=(4,4,4) )
        self.assertTrue( approx_equal( v , a1*a2*a3 ))

        z = grid.depth( ijk = (4,4,4 ))
        self.assertTrue( approx_equal( z , 4.5*a3 ))

        g1 = grid.global_index( ijk = (2,2,2) )
        g2 = grid.global_index( ijk = (4,4,4) )
        (dx,dy,dz) = grid.distance( g2 , g1 )
        self.assertTrue( approx_equalv([dx,dy,dz],[2*a1,2*a2,2*a3] ) )

        self.assertTrue( grid.cell_contains(2.5*a1 , 2.5*a2, 2.5*a3 , ijk=(2,2,2)))

        ijk = grid.find_cell( 1.5*a1 , 2.5*a2, 3.5*a3 )
        self.assertTrue( approx_equalv( ijk , [1 , 2 , 3]))



    def testCreate(self):
        grid = self.create( grdecl_file() )
        self.assertTrue( grid )


    def testACTNUM(self):
        g1 = self.create( grdecl_file() )
        g2 = self.create( grdecl_file() , load_actnum = False )
        self.assertTrue( g1.equal( g2 ) )


    def testTime(self):
        t0 = time.clock()
        g1 = ecl.EclGrid( egrid_file() )
        t1 = time.clock()
        t = t1 - t0
        self.assertTrue( t < 1.0 )


    def testSave(self):
        work_area = TestArea("python/grid-test/testSave" , True)
        g1 = ecl.EclGrid( egrid_file() )

        g1.save_EGRID( "test.EGRID" )
        g2 = ecl.EclGrid( "test.EGRID" )
        self.assertTrue( g1.equal( g2 ) )

        g1.save_GRID( "test.GRID" )
        g2 = ecl.EclGrid( "test.GRID" )
        self.assertTrue( g1.equal( g2 ) )
        
        fileH = open("test.grdecl" , "w")                                                  
        g1.save_grdecl( fileH )
        fileH.close()
        g2 = self.create( "test.grdecl" )
        self.assertTrue( g1.equal( g2 ) )


    def testCoarse(self):
        work_area = TestArea("python/grid-test/testCoarse" )
        testGRID = True
        g1 = ecl.EclGrid( test_path("test-data/Statoil/ECLIPSE/LGCcase/LGC_TESTCASE2.EGRID") )
        
        g1.save_EGRID( "LGC.EGRID" )
        g2 = ecl.EclGrid( "LGC.EGRID")
        self.assertTrue( g1.equal(g2, verbose = True) )
        
        if testGRID:
            g1.save_GRID( "LGC.GRID" )
            g3 = ecl.EclGrid( "LGC.GRID")
            self.assertTrue( g1.equal(g3 , verbose = True) )

        self.assertTrue( g1.coarse_groups() == 3384)


    def testRaiseIOError(self):
        self.assertRaises( IOError , load_does_not_exist )



    def testDual(self):
        work_area = TestArea("python/grid-test/testDual" , True)
        grid = ecl.EclGrid( egrid_file() )
        self.assertFalse( grid.dual_grid )
        self.assertTrue( grid.nactive_fracture == 0 )
        
        grid2 = ecl.EclGrid( grid_file() )
        self.assertFalse( grid.dual_grid )
        self.assertTrue( grid.nactive_fracture == 0 )
        
        dgrid = ecl.EclGrid( test_path("test-data/Statoil/ECLIPSE/DualPoro/DUALPOR_MSW.EGRID") )
        self.assertTrue( dgrid.nactive == dgrid.nactive_fracture )
        self.assertTrue( dgrid.nactive == 46118 )

        dgrid2 = ecl.EclGrid( test_path("test-data/Statoil/ECLIPSE/DualPoro/DUALPOR_MSW.GRID") )
        self.assertTrue( dgrid.nactive == dgrid.nactive_fracture )
        self.assertTrue( dgrid.nactive == 46118 )        
        self.assertTrue( dgrid.equal( dgrid2 ))
        
        
        # The DUAL_DIFF grid has been manipulated to create a
        # situation where some cells are only matrix active, and some
        # cells are only fracture active.
        dgrid = ecl.EclGrid( test_path("test-data/Statoil/ECLIPSE/DualPoro/DUAL_DIFF.EGRID") )
        self.assertTrue( dgrid.nactive == 106 )
        self.assertTrue( dgrid.nactive_fracture == 105 )

        self.assertTrue( dgrid.get_active_fracture_index( global_index = 0 ) == -1 )
        self.assertTrue( dgrid.get_active_fracture_index( global_index = 2 ) == -1 )
        self.assertTrue( dgrid.get_active_fracture_index( global_index = 3 ) ==  0 )
        self.assertTrue( dgrid.get_active_fracture_index( global_index = 107 ) ==  104)

        self.assertTrue( dgrid.get_active_index( global_index = 1 )   == 1)
        self.assertTrue( dgrid.get_active_index( global_index = 105 ) == 105)
        self.assertTrue( dgrid.get_active_index( global_index = 106 ) == -1)
        self.assertTrue( dgrid.get_global_index1F( 2 ) == 5 )

        dgrid.save_GRID("DUAL_DIFF.GRID")
        dgrid2 = ecl.EclGrid( "DUAL_DIFF.GRID" )
        self.assertTrue( dgrid.equal( dgrid2 ))
        

        
        




def fast_suite():
    suite = unittest.TestSuite()
    suite.addTest( GridTest( 'testDual' ))
    suite.addTest( GridTest( 'testGRID' ))
    suite.addTest( GridTest( 'testEGRID' ))
    suite.addTest( GridTest( 'testCreate' ))
    
    suite.addTest( GridTest( 'testSave' ))
    suite.addTest( GridTest( 'testTime' ))
    suite.addTest( GridTest( 'testACTNUM') )
    suite.addTest( GridTest( 'testRect' ))
    suite.addTest( GridTest( 'testCoarse' ))
    suite.addTest( GridTest( 'testRaiseIOError' ))
    return suite

def test_suite(argv):
    return fast_suite()


if __name__ == "__main__":
    unittest.TextTestRunner().run( fast_suite() )
